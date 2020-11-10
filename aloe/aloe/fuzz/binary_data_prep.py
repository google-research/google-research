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
from aloe.common.configs import cmd_args
import pickle as cp
import os
import sys
import random
import numpy as np
import binascii
from tqdm import tqdm

def get_common_prefix(prefix, new_st):
    if prefix is None:
        return new_st
    if len(prefix) == 0 or len(new_st) == 0:
        return ''
    if prefix[0] != new_st[0]:
        return ''
    l = 1
    r = min(len(prefix), len(new_st))
    ans = None
    while l <= r:
        mid = (l + r) // 2
        if prefix[:mid] == new_st[:mid]:
            l = mid + 1
            ans = mid
        else:
            r = mid - 1
    return prefix[:ans]


def save_xfix(xfix, out_name):
    b_list = []
    for i in range(0, len(xfix), 2):
        b = int(xfix[i : i + 2], 16)
        b_list.append(b)

    xfix = np.array(b_list, dtype=np.int32)
    np.save(out_name, xfix)


def file_gen(root_folder):
    for fname in os.listdir(root_folder):
        cur_fname = os.path.join(root_folder, fname)
        if os.path.isdir(cur_fname):
            for sub_files in file_gen(cur_fname):
                yield sub_files
        else:
            yield cur_fname


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)

    prefix = None
    suffix = None
    hex_stream = []
    list_raw = []
    for fname in tqdm(file_gen(cmd_args.data_dir)):
        with open(fname, 'rb') as f:
            hexdata = binascii.hexlify(f.read())

        b_list = []
        for i in range(0, len(hexdata), 2):
            b = int(hexdata[i : i + 2], 16)
            b_list.append(b)

        for i in range(len(b_list) - cmd_args.window_size):
            hex_stream.append([i] + b_list[i : i + cmd_args.window_size])
        list_raw.append(np.array(b_list, dtype=np.int32))
        prefix = get_common_prefix(prefix, hexdata)
        suffix = get_common_prefix(suffix, hexdata[::-1])

    hex_stream = np.array(hex_stream, dtype=np.int32)
    print('# samples', hex_stream.shape[0])
    out_data = os.path.join(cmd_args.save_dir, 'hex_stream.npy')
    np.save(out_data, hex_stream)

    save_xfix(prefix, os.path.join(cmd_args.save_dir, 'prefix.npy'))

    suffix = suffix[::-1]
    save_xfix(suffix, os.path.join(cmd_args.save_dir, 'suffix.npy'))

    raw_data = os.path.join(cmd_args.save_dir, 'raw.pkl')
    with open(raw_data, 'wb') as f:
        cp.dump(list_raw, f, cp.HIGHEST_PROTOCOL)
