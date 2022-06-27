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
import os
import ctypes
import sys
import numpy as np
import torch
from lego.cpp_sampler import libsampler
import torch.multiprocessing as mp
import torch.distributed as dist
from collections import defaultdict
from tqdm import tqdm

dll_path = '%s/build/dll/libsampler_c.so' % os.path.dirname(os.path.realpath(__file__))

class _sampler_clib(object):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.lib = ctypes.CDLL('%s/build/dll/libsampler_c.so' % dir_path)

if os.path.exists(dll_path):
    SCLIB = _sampler_clib()
else:
    SCLIB = None


def create_kg(num_ent, num_rel, dtype):
    if dtype == 'uint32':
        return libsampler.KG32(num_ent, num_rel)
    else:
        assert dtype == 'uint64'
        return libsampler.KG64(num_ent, num_rel)

def naive_sampler(dtype):
    if dtype == 'uint32':
        return libsampler.NaiveSampler32
    else:
        assert dtype == 'uint64'
        return libsampler.NaiveSampler64

def no_search_sampler(dtype):
    if dtype == 'uint32':
        return libsampler.NoSearchSampler32
    else:
        assert dtype == 'uint64'
        return libsampler.NoSearchSampler64

def rejection_sampler(dtype):
    if dtype == 'uint32':
        return libsampler.RejectionSampler32
    else:
        assert dtype == 'uint64'
        return libsampler.RejectionSampler64

def test_sampler(dtype):
    if dtype == 'uint32':
        return libsampler.TestSampler32
    else:
        assert dtype == 'uint64'
        return libsampler.TestSampler64


class KGMem(object):
    def __init__(self, dtype, mem=None):
        self.mem = mem
        self.dtype = dtype

    def load(self, fname):
        fsize = os.stat(fname).st_size
        if self.dtype == 'uint32':
            assert fsize % 4 == 0
            n_int = fsize // 4
            self.mem = torch.IntTensor(n_int)
        else:
            assert fsize % 8 == 0
            n_int = fsize // 8
            self.mem = torch.LongTensor(n_int)
        SCLIB.lib.load_binary_file(ctypes.c_char_p(fname.encode()),
                                   ctypes.c_void_p(self.mem.data_ptr()),
                                   ctypes.c_ulonglong(n_int),
                                   ctypes.c_char_p(self.dtype.encode()))

    def create_kg(self):
        assert self.mem is not None
        kg = libsampler.KG32() if self.dtype == 'uint32' else libsampler.KG64()
        SCLIB.lib.load_kg_from_binary(ctypes.c_void_p(kg.ptr()),
                                      ctypes.c_void_p(self.mem.data_ptr()),
                                      ctypes.c_ulonglong(self.mem.shape[0]),
                                      ctypes.c_char_p(self.dtype.encode()))
        return kg

    def share_memory(self):
        assert self.mem is not None
        self.mem.share_memory_()


def load_kg_from_numpy(kg, triplets, has_reverse_edges=False):
    assert triplets.dtype == np.int64
    max_id = np.max(triplets[:, 0])
    max_id = max(max_id, np.max(triplets[:, 2]))
    max_rel = np.max(triplets[:, 1])
    assert max_id < kg.num_ent
    if has_reverse_edges:
        assert max_rel < kg.num_rel
    else:
        assert max_rel * 2 < kg.num_rel
    SCLIB.lib.load_kg_from_numpy(ctypes.c_void_p(kg.ptr()),
                                 ctypes.c_void_p(triplets.ctypes.data),
                                 ctypes.c_ulonglong(triplets.shape[0]),
                                 ctypes.c_bool(has_reverse_edges),
                                 ctypes.c_char_p(kg.dtype.encode()))


def test_kg(rank, kg_mem):
    dist.barrier()
    print('rank', rank)


if __name__ == '__main__':
    kg_mem = KGMem()
    db_name = 'FB15k'
    data_folder = os.path.join(os.path.expanduser('~'), 'data/knowledge_graphs/%s' % db_name)
    kg_mem.load(data_folder + '/train_bidir.bin')
    # kg_mem.share_memory()
    kg = kg_mem.create_kg()
    print('num ent', kg.num_ent)
    print('num rel', kg.num_rel)
    print('num edges', kg.num_edges)
    # mp.set_start_method('spawn')

    # procs = []
    # for i in range(4):
    #     proc = mp.Process(target=test_kg, args=(i, kg_mem))
    #     proc.start()
    #     procs.append(proc)
    # for proc in procs:
    #     proc.join()
