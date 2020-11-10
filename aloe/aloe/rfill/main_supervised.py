# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

from aloe.common.configs import cmd_args, set_device
from aloe.rfill.utils.dataset import Dataset, RawStaticRfill, CookedInfRfill
from aloe.rfill.utils.rfill_parser import RFillNode, evaluate_prog
from aloe.rfill.sampler import RfillRnnSampler
from aloe.rfill.seq_embed import BidirIOEmbed, MLPIOEmbed, TripletIOEmbed
from itertools import chain
from tqdm import tqdm


def load_model(epoch_load):
    encoder_dump = os.path.join(cmd_args.save_dir, 'encoder-%d.ckpt' % epoch_load)
    print('loading encoder from', encoder_dump)
    encoder.load_state_dict(torch.load(encoder_dump, map_location=cmd_args.device))

    decoder_dump = os.path.join(cmd_args.save_dir, 'decoder-%d.ckpt' % epoch_load)
    print('loading decoder from', decoder_dump)
    decoder.load_state_dict(torch.load(decoder_dump, map_location=cmd_args.device), strict=False)


def test_passed(inputs, outputs, prog):
    passed = True
    for x, y in zip(inputs, outputs):
        out = evaluate_prog(prog, x)
        if y != out:
            passed = False
            break
    return passed


def test_topk(test_db, eval_func, epoch_load=None):
    if epoch_load is not None and epoch_load >= 0:
        load_model(epoch_load)
    test_gen = DataLoader(test_db, batch_size=cmd_args.batch_size, shuffle=False,
                          collate_fn=test_db.collate_fn, num_workers=cmd_args.num_proc, drop_last=False)

    pbar = tqdm(test_gen)
    acc = 0.0
    num_done = 0
    for cur_samples in pbar:
        list_inputs, list_outputs, _, _ = cur_samples
        public_inputs, private_inputs = [x[:cmd_args.numPublicIO] for x in list_inputs], [x[cmd_args.numPublicIO:] for x in list_inputs]
        public_outputs, private_outputs = [x[:cmd_args.numPublicIO] for x in list_outputs], [x[cmd_args.numPublicIO:] for x in list_outputs]

        _, list_progs, sizes, _ = eval_func(public_inputs, public_outputs)
        offset = 0
        for i, s in enumerate(sizes):
            eval_prog = None
            k_cnt = 0
            cur_passed = False
            for j in range(s):
                prog = list_progs[offset + j]
                prog = RFillNode.from_tokens(prog)
                assert prog is not None
                if not test_passed(public_inputs[i], public_outputs[i], prog):
                    continue
                k_cnt += 1
                cur_passed = test_passed(private_inputs[i], private_outputs[i], prog)
                if cur_passed:
                    break
                if k_cnt >= cmd_args.eval_topk:
                    break
            acc += cur_passed
            offset += s
        num_done += len(list_inputs)
        pbar.set_description('frac: %.2f, acc: %.2f' % (num_done / test_db.num_programs, acc / num_done))
    return acc / test_db.num_programs, 1.0


if __name__ == '__main__':
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    set_device(cmd_args.gpu)

    # build model
    if cmd_args.io_enc == 'rnn':
        encoder = BidirIOEmbed(cmd_args)
    elif cmd_args.io_enc.startswith('mlp'):
        n_hidden = 2 if cmd_args.io_enc == 'mlp' else int(cmd_args.io_enc.split('-')[-1])
        encoder = MLPIOEmbed(cmd_args, n_hidden)
    elif cmd_args.io_enc == 'triplet':
        encoder = TripletIOEmbed(cmd_args)
    else:
        raise NotImplementedError
    encoder = encoder.to(cmd_args.device)
    decoder = RfillRnnSampler(cmd_args).to(cmd_args.device)
    if cmd_args.epoch_load is not None:
        load_model(cmd_args.epoch_load)

    if cmd_args.phase == 'test':
        test_json = os.path.join(cmd_args.data_dir, 'test.json')
        test_db = RawStaticRfill(cmd_args, test_json)
        eval_func = lambda i, o: decoder(encoder(i, o)[-1], gen_method=cmd_args.eval_method)
        acc, valid = test_topk(test_db, eval_func, epoch_load=cmd_args.epoch_load)
        print('accuracy: %.4f' % acc)
        print('valid: %.4f' % valid)
        sys.exit()

    train_db = CookedInfRfill(cmd_args,
                                io2idx_func=encoder.io2idx_func,
                                fn_pad_in=encoder.padded_inputs,
                                fn_pad_out=encoder.padded_outputs,
                                need_mask=cmd_args.masked)

    train_load = DataLoader(train_db, batch_size=cmd_args.batch_size, shuffle=False,
                            collate_fn=train_db.collate_fn, num_workers=cmd_args.num_proc, drop_last=False)
    train_gen = iter(train_load)
    optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=cmd_args.learning_rate)

    for epoch in range(cmd_args.num_epochs):
        pbar = tqdm(range(train_db.iter_per_epoch))
        for it in pbar:
            try:
                cur_samples = next(train_gen)
            except StopIteration:
                train_gen = iter(train_load)
                cur_samples = next(train_gen)
            list_inputs, list_outputs, expr_list, cooked_data = cur_samples
            optimizer.zero_grad()
            _, _, io_embed = encoder(list_inputs, list_outputs, cooked_data=cooked_data)

            ll, _ = decoder(io_embed, expr_list=expr_list, cooked_data=cooked_data)
            loss = -torch.mean(ll)
            loss.backward()
            if cmd_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(chain(encoder.parameters(), decoder.parameters()), max_norm=cmd_args.grad_clip)
            optimizer.step()

            pbar.set_description('epoch: %.2f, loss: %.2f' % (epoch + it / train_db.iter_per_epoch, loss.item()))

        if epoch % cmd_args.epoch_save == 0:
            torch.save(encoder.state_dict(), os.path.join(cmd_args.save_dir, 'encoder-%d.ckpt' % epoch))
            torch.save(decoder.state_dict(), os.path.join(cmd_args.save_dir, 'decoder-%d.ckpt' % epoch))
