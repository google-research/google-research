# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
import os
import sys
import random
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from aloe.common.configs import cmd_args, set_device
from aloe.common.sampler import GibbsSampler, gibbs_step
from aloe.fuzz.main_varlen import get_score_func, get_editor
from aloe.fuzz.utils import HexStreamDataset, to_hex
from tqdm import tqdm
import binascii


def rand_modify(init_samples):
    w_size = cmd_args.window_size // 2
    sample_lens = [x.shape[0] for x in init_samples]
    for _ in range(cmd_args.num_change):
        pos_list = []
        cur_samples = []
        list_start = []
        for i in range(len(sample_lens)):
            start_pos = np.random.randint(0, sample_lens[i] - cmd_args.window_size)
            list_start.append(start_pos)

            cur_poses = torch.LongTensor(list(range(start_pos, start_pos + cmd_args.window_size))).to(cmd_args.device).view(1, -1)
            pos_list.append(cur_poses)

            frag_bytes = init_samples[i][start_pos:start_pos + cmd_args.window_size].view(1, -1)
            cur_samples.append(frag_bytes)

        for r in range(cmd_args.gibbs_rounds):
            for i in range(len(sample_lens)):
                cur_score_fn = lambda samples: score_func(pos_list[i], samples)
                pos = np.random.randint(cmd_args.window_size)
                cur_samples[i] = gibbs_step(cur_samples[i], pos, 256, cur_score_fn)

        for i in range(len(sample_lens)):
            start_pos = list_start[i]
            init_samples[i][start_pos:start_pos + cmd_args.window_size] = cur_samples[i]

    pngs = []
    for i in range(len(init_samples)):
        pngs.append(to_hex(init_samples[i].data.cpu().numpy()))
    return pngs


if __name__ == '__main__':
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    torch.set_grad_enabled(False)

    score_func = get_score_func(cmd_args)
    gibbs_sampler = GibbsSampler(n_choices=256, discrete_dim=cmd_args.window_size, device=cmd_args.device)
    if cmd_args.epoch_load is not None:
        model_dump = os.path.join(cmd_args.save_dir, 'model-%d.ckpt' % cmd_args.epoch_load)
        print('loading model dump from', model_dump)
        ckpt = torch.load(model_dump, map_location=cmd_args.device)
        score_func.load_state_dict(ckpt['score_func'])

    db = HexStreamDataset(cmd_args)
    data_load = DataLoader(db, batch_size=cmd_args.batch_size, shuffle=True,
                            collate_fn=db.collate_fn, num_workers=cmd_args.num_proc, drop_last=True)

    data_name = cmd_args.data_dir.split('/')[-1].split('-')[0]
    out_dir = os.path.join(cmd_args.save_dir, 'aloe-%s-num-%d-c-%d' % (data_name, cmd_args.num_gen, cmd_args.num_change))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    cmd_args.batch_size = 1

    for st in tqdm(range(0, cmd_args.num_gen, cmd_args.batch_size)):
        num = min(cmd_args.batch_size, cmd_args.num_gen - st)
        init_samples = []
        for _ in range(num):
            cur_sample = np.random.choice(db.list_raw)
            init_samples.append(torch.LongTensor(cur_sample).view(-1).clone().to(cmd_args.device))
        cur_samples = rand_modify(init_samples)

        for idx in range(st, st + num):
            fname = os.path.join(out_dir, 'sample-%d.png' % idx)

            with open(fname, 'wb') as fout:
                fout.write(binascii.unhexlify(cur_samples[idx - st]))
