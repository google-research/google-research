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
import numpy as np
import os
import pickle as cp
import sys
import random
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch_scatter import scatter_add

from aloe.common.configs import cmd_args, set_device
from aloe.rfill.utils.dataset import Dataset, RawStaticRfill, CookedInfRfill
from aloe.rfill.utils.rfill_parser import RFillNode, evaluate_prog
from aloe.rfill.editor import RfillSampler, shortest_rand_edit, perform_edit
from tqdm import tqdm
from aloe.rfill.main_supervised import test_topk, test_passed


def check_prog(pred_prog, gt_prog, list_inputs, list_outputs):
    prog = RFillNode.from_tokens(pred_prog)
    if prog is None:
        return 0, 0, 0
    passed = True
    for x, y in zip(list_inputs, list_outputs):
        out = evaluate_prog(prog, x)
        if y != out:
            passed = False
            break
    if passed:
        #same = ''.join(pred_prog) == ''.join(gt_prog)
        same = False
        return 1, 1, same
    else:
        return 1, 0, 0


def load_pretrained(sampler):
    encoder_dump = os.path.join(cmd_args.save_dir, 'encoder-%d.ckpt' % -1)
    print('loading pretrained encoder from', encoder_dump)
    sampler.encoder.load_state_dict(torch.load(encoder_dump, map_location=cmd_args.device))

    decoder_dump = os.path.join(cmd_args.save_dir, 'decoder-%d.ckpt' % -1)
    print('loading pretrained decoder from', decoder_dump)
    sampler.q0.load_state_dict(torch.load(decoder_dump, map_location=cmd_args.device), strict=False)


def get_init_progs(sampler, cur_samples, cmd_args):
    raw_io_embed = sampler.forward_io(cur_samples)
    io_embed = raw_io_embed.repeat(cmd_args.num_importance_samples, 1)
    if cmd_args.inf_type == 'argmax':
        _, init_progs, _, _ = sampler.forward_q0(raw_io_embed, gen_method='argmax')
        init_progs = init_progs * cmd_args.num_importance_samples
    elif cmd_args.inf_type == 'sample':
        _, init_progs, _, _ = sampler.forward_q0(io_embed, gen_method='sample')
    else:
        _, raw_init_progs, sizes, _ = sampler.forward_q0(io_embed, gen_method=cmd_args.inf_type)
        init_progs = []
        offset = 0
        public_inputs, public_outputs = [x[:cmd_args.numPublicIO] for x in list_inputs], [x[:cmd_args.numPublicIO] for x in list_outputs]
        for i, s in enumerate(sizes):
            the_prog = raw_init_progs[offset]
            for j in range(s):
                prog = raw_init_progs[offset + j]
                prog = RFillNode.from_tokens(prog)
                assert prog is not None
                if test_passed(public_inputs[i], public_outputs[i], prog):
                    the_prog = raw_init_progs[offset + j]
                    break
            init_progs.append(the_prog)
            offset += s
    return io_embed, init_progs


def get_step_probs(sampler, list_inputs, list_outputs, io_embed, init_progs, target_progs, need_check=True):
    expr_list = []
    pos_list = []
    subexpr_list = []
    scatter_idx = []
    proposal_dist = []
    for i, (x, y) in enumerate(zip(init_progs, target_progs)):
        if need_check:
            is_acc = check_prog(x, y, list_inputs[i], list_outputs[i])[1]  # no need to modify if it is accurate
            if is_acc:
                log_prob = 0
                diff_list = []
        else:
            is_acc = False
        if not is_acc:
            _, log_prob, diff_list = shortest_rand_edit(x, y)
        proposal_dist.append(log_prob)
        e1 = x
        for diff in diff_list:
            pos_list.append(diff[0])
            subexpr_list.append(diff[1])
            expr_list.append(e1)
            e1 = perform_edit(e1, diff[0], diff[1])
        expr_list.append(e1)
        pos_list.append(len(e1))
        scatter_idx += [i] * (1 + len(diff_list))
        subexpr_list.append(None)
    proposal_dist = torch.tensor(proposal_dist).view(-1, 1).to(cmd_args.device)
    scatter_idx = torch.LongTensor(scatter_idx).to(cmd_args.device)

    ll_step = sampler.forward_onestep(io_embed[scatter_idx], expr_list, pos_list=pos_list, subexpr_list=subexpr_list)
    ll_step = scatter_add(ll_step, scatter_idx, 0, dim_size=io_embed.shape[0])
    return ll_step, proposal_dist


if __name__ == '__main__':
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    set_device(cmd_args.gpu)

    test_json = os.path.join(cmd_args.data_dir, 'test.json')
    test_db = RawStaticRfill(cmd_args, test_json)

    sampler = RfillSampler(cmd_args).to(cmd_args.device)
    ckpt = None
    assert cmd_args.epoch_load is not None
    if cmd_args.epoch_load < 0:
        load_pretrained(sampler)
        eval_func = lambda i, o: sampler((i, o, None, None), 0, gen_method=cmd_args.eval_method)
        acc, _ = test_topk(test_db, eval_func)
        print('init accuracy: %.4f' % acc)
    else:
        dump = os.path.join(cmd_args.save_dir, 'sampler-%d.ckpt' % cmd_args.epoch_load)
        print('loading sampler from', dump)
        ckpt = torch.load(dump, map_location=cmd_args.device)
        sampler.load_state_dict(ckpt['model'])

    if cmd_args.phase == 'test':
        eval_func = lambda i, o: sampler((i, o, None, None), 10, gen_method=cmd_args.eval_method)
        acc, valid = test_topk(test_db, eval_func)
        print('test accuracy: %.4f' % acc)
        print('test valid: %.4f' % valid)
        sys.exit()

    if cmd_args.phase == 'plot':
        test_gen = DataLoader(test_db, batch_size=1, shuffle=False,
                            collate_fn=test_db.collate_fn, num_workers=cmd_args.num_proc, drop_last=False)
        pbar = tqdm(test_gen)
        num_samples = 0
        num_mod = 0.0
        tot_init = 0.0
        tot_final = 0.0
        list_plot = []
        for cur_samples in pbar:
            list_inputs, list_outputs, _, _ = cur_samples
            public_inputs, private_inputs = [x[:cmd_args.numPublicIO] for x in list_inputs], [x[cmd_args.numPublicIO:] for x in list_inputs]
            public_outputs, private_outputs = [x[:cmd_args.numPublicIO] for x in list_outputs], [x[cmd_args.numPublicIO:] for x in list_outputs]
            traj = sampler((public_inputs, public_outputs, None, None), 10, gen_method=cmd_args.eval_method, phase='plot')
            num_samples += 1
            num_mod += len(traj)
            init_acc = check_prog(traj[0][0], [], list_inputs[0], list_outputs[0])[1]
            if len(traj) > 1:
                final_acc = check_prog(traj[-1][0], [], list_inputs[0], list_outputs[0])[1]
            else:
                final_acc = init_acc
            if final_acc > init_acc:
                list_plot.append((traj[0][0], traj[-1][0], list_inputs[0], list_outputs[0]))
            tot_init += init_acc
            tot_final += final_acc
            pbar.set_description('avg steps %.2f, init acc %.2f, final acc %.2f' % (num_mod / num_samples, tot_init / num_samples, tot_final / num_samples))
        with open(os.path.join(cmd_args.save_dir, 'edit_plot-%d.pkl' % cmd_args.epoch_load), 'wb') as f:
            cp.dump(list_plot, f, cp.HIGHEST_PROTOCOL)
        sys.exit()

    train_db = CookedInfRfill(cmd_args,
                            io2idx_func=sampler.encoder.io2idx_func,
                            fn_pad_in=sampler.encoder.padded_inputs,
                            fn_pad_out=sampler.encoder.padded_outputs,
                            need_mask=cmd_args.masked)
    train_load = DataLoader(train_db, batch_size=cmd_args.batch_size, shuffle=False,
                            collate_fn=train_db.collate_fn, num_workers=cmd_args.num_proc, drop_last=False)
    train_gen = iter(train_load)

    optimizer = optim.Adam(sampler.parameters(), lr=cmd_args.learning_rate)
    if ckpt is not None:
        optimizer.load_state_dict(ckpt['opt_state'])
    epoch_start = 0 if ckpt is None else ckpt['epoch'] + 1
    for epoch in range(epoch_start, cmd_args.num_epochs):
        pbar = tqdm(range(train_db.iter_per_epoch))
        for it in pbar:
            try:
                cur_samples = next(train_gen)
            except StopIteration:
                train_gen = iter(train_load)
                cur_samples = next(train_gen)
            optimizer.zero_grad()

            with torch.no_grad():
                list_inputs, list_outputs, target_progs, _ = cur_samples
                list_inputs = list_inputs * cmd_args.num_importance_samples
                list_outputs = list_outputs * cmd_args.num_importance_samples
                target_progs = target_progs * cmd_args.num_importance_samples

                io_embed, init_progs = get_init_progs(sampler, cur_samples, cmd_args)

            ll_step, proposal_dist = get_step_probs(sampler, list_inputs, list_outputs, io_embed, init_progs, target_progs)

            with torch.no_grad():
                log_ratio = (ll_step - proposal_dist).view(cmd_args.num_importance_samples, -1)
                weight = F.softmax(log_ratio, dim=0).view(ll_step.shape)
            ll_step = ll_step * weight
            loss = -torch.mean(ll_step) * cmd_args.num_importance_samples
            loss.backward()
            if cmd_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(sampler.parameters(), max_norm=cmd_args.grad_clip)
            optimizer.step()

            pbar.set_description('epoch: %d, g: %.2f' % (epoch, loss.item()))

        if epoch % cmd_args.epoch_save == 0:
            save_dict = {
                'epoch': epoch,
                'model': sampler.state_dict(),
                'opt_state': optimizer.state_dict()
            }
            torch.save(save_dict, os.path.join(cmd_args.save_dir, 'sampler-%d.ckpt' % epoch))
