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

from typing import Callable

from acme import specs

from jrl.agents import batch_ensemble_msg
from jrl.agents import bc
from jrl.agents import cql
from jrl.agents import msg
from jrl.agents import snr
from jrl.utils import networks
from jrl.utils.agent_utils import RLComponents


def create_agent(
    algorithm,
    spec,
    # get_param: Callable[[str], Any],
    create_data_iter_fn,
    # counter_prefix: str = 'learner_',
    # gin_context: Optional[GINContext] = None,
    logger_fn,
):
  if algorithm == 'bc':
    return bc.BCRLComponents(logger_fn, spec, create_data_iter_fn)
  elif algorithm == 'cql':
    return cql.CQLRLComponents(logger_fn, spec, create_data_iter_fn)
  elif algorithm == 'msg':
    return msg.MSGRLComponents(logger_fn, spec, create_data_iter_fn)
  elif algorithm == 'batch_ensemble_msg':
    return batch_ensemble_msg.BatchEnsembleMSGRLComponents(
        logger_fn, spec, create_data_iter_fn)
  elif algorithm == 'snr':
    return snr.SNRRLComponents(logger_fn, spec, create_data_iter_fn)
  else:
    raise NotImplementedError()
