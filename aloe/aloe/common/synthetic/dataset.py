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
from aloe.common.synthetic.toy_data_gen import inf_train_gen


class ToyDataset(object):

    def __init__(self, dim, data_file=None, static_data=None):
        if data_file is not None:
            self.static_data = np.load(data_file)
        elif static_data is not None:
            self.static_data = static_data
        else:
            self.static_data = None
        self.dim = dim

    def gen_batch(self, batch_size):
        raise NotImplementedError

    def data_gen(self, batch_size, auto_reset):
        if self.static_data is not None:
            num_obs = self.static_data.shape[0]
            while True:
                for pos in range(0, num_obs, batch_size):
                    if pos + batch_size > num_obs:  # the last mini-batch has fewer samples
                        if auto_reset:  # no need to use this last mini-batch
                            break
                        else:
                            num_samples = num_obs - pos
                    else:
                        num_samples = batch_size
                    yield self.static_data[pos : pos + num_samples, :]
                if not auto_reset:
                    break
                np.random.shuffle(self.static_data)
        else:
            while True:
                yield self.gen_batch(batch_size)


class OnlineToyDataset(ToyDataset):
    def __init__(self, data_name):
        super(OnlineToyDataset, self).__init__(2)
        self.data_name = data_name
        self.rng = np.random.RandomState()

        rng = np.random.RandomState(1)
        samples = inf_train_gen(self.data_name, rng, 5000)
        self.f_scale = np.max(np.abs(samples)) + 1
        self.int_scale = 2 ** 15 / (self.f_scale + 1)
        print('f_scale,', self.f_scale, 'int_scale,', self.int_scale)

    def gen_batch(self, batch_size):
        return inf_train_gen(self.data_name, self.rng, batch_size)
