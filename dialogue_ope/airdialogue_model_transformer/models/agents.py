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

"""Agents for handling the generation of Airdialogue.
"""
from itertools import chain
from functools import lru_cache

import torch as th
import numpy as np

from parlai.utils.torch import padded_tensor
from parlai.utils.misc import round_sigfigs

from parlai.agents.transformer.transformer import TransformerGeneratorAgent

from .modules import EndToEndModel
from .utils import INTENT_DICT, STATUS_DICT, intent_to_status

from copy import deepcopy

from airdialogue.evaluator.evaluator_main import action_obj_to_str
from airdialogue.prepro.tokenize_lib import tokenize_kb
from airdialogue.evaluator.selfplay_utils import compute_reward

DEFAULT_OPTS = {
    'learningrate': 5e-4,
    'optimizer': 'adam',
    'lr_scheduler': 'invsqrt',
    'warmup_updates': 5000,
    'clip_norm': 0.1,
    'ffn_size': 512,
    'embedding_size': 256,
    'n_heads': 2,
    'dropout': 0.2,
    'n_layers': 5,
    'n_layers_knowledge': 2,
    'betas': '0.9,0.98',
    'truncate': 512,
    'n_positions': 512,
    'add_token_knowledge': True,
    'dict_textfields': 'text,labels,action_name,intent,tickets,reservation',
}


class _GenericAirAgent(TransformerGeneratorAgent):

  @classmethod
  def add_cmdline_args(cls, argparser):
    argparser.set_defaults(**DEFAULT_OPTS)
    super(_GenericAirAgent, cls).add_cmdline_args(argparser)

  def batchify(self, obs_batch):
    batch = super().batchify(obs_batch)
    reordered_observations = [obs_batch[i] for i in batch.valid_indices]
    return batch

  def eval_step(self, batch):
    output = super().eval_step(batch)
    if batch.return_encoder_state:
      model = self.model
      model.eval()
      if isinstance(model, th.nn.parallel.DistributedDataParallel):
        model = self.model.module
      encoder_states = model.encoder(*self._encoder_input(batch))
      encoder_states = list(zip(*encoder_states))
      output['encoder_states'] = encoder_states
    return output

  def batchify(self, obs_batch):
    batch = super().batchify(obs_batch)
    # the following works with eval_step to return encoder_state in evaluation
    batch['return_encoder_state'] = False
    if 'return_encoder_state' in obs_batch[0]:
      if obs_batch[0]['return_encoder_state']:
        batch['return_encoder_state'] = True
    return batch


class AgentAgent(_GenericAirAgent):

  def __init__(self, opt, shared=None):
    super().__init__(opt, shared)
    self._vectorize_text = lru_cache(int(2**20))(self._vectorize_text)
    self.name_vec_len = opt.get('name_vec_len')
    self.wei_intent = opt.get('wei_intent')
    self.wei_name = opt.get('wei_name')
    self.wei_flight = opt.get('wei_flight')
    self.id = 'Agent'

  def _dummy_batch(self, bsz, maxlen):
    batch = super()._dummy_batch(bsz, maxlen)
    batch['know_vec'] = th.ones(bsz, 2, 2).long().cuda()
    batch['res_vec'] = th.ones(bsz, 2).long().cuda()
    # bool/uint8 backwards for pytorch 1.0/1.2 compatibility
    ck_mask = (th.ones(bsz, 2, dtype=th.uint8) != 0).cuda()
    batch['ck_mask'] = ck_mask
    batch['intent_ids'] = th.zeros(bsz).long().cuda()
    batch['status_ids'] = th.zeros(bsz).long().cuda()
    batch['flight_ids'] = th.zeros(bsz).long().cuda()
    batch['name_vec'] = th.ones(bsz, self.name_vec_len).long().cuda()
    batch['is_outputact'] = th.zeros(bsz).bool().cuda()
    batch['has_res'] = th.zeros(bsz).bool().cuda()
    return batch

  def get_air_score(self, encoder_states, expected_action, kb):
    intent_out = encoder_states[3]
    intent_pred = intent_out.max(-1)[1].item()
    name_out = encoder_states[4]
    name_pred_vec = name_out.max(-1)[1]
    name_pred_vec_stop = 0
    for i in name_pred_vec:
      if i == 0:
        break
      else:
        name_pred_vec_stop += 1
    name_pred_vec = name_pred_vec[:name_pred_vec_stop]
    name_pred = self._v2t(name_pred_vec)
    name_pred = ' '.join([i.capitalize() for i in name_pred.split(' ')])
    ticket_attn = encoder_states[2]
    ticket_pred = ticket_attn.max(-1)[1].item()
    status_pred, flight_pred = intent_to_status(ticket_pred,
                                                kb['reservation'] > 0,
                                                intent_pred)

    if flight_pred == 0:
      flight_pred = []
    else:
      flight_pred = [flight_pred - 1 + 1000]

    pred = {
        'status': STATUS_DICT[status_pred],
        'flight': flight_pred,
        'name': name_pred,
    }
    reward, name_score, flight_score, status_score = compute_reward(
        action_obj_to_str(pred),
        action_obj_to_str(expected_action),
        tokenize_kb(kb),
    )
    pred['reward'] = reward
    pred['name_score'] = name_score
    pred['flight_score'] = flight_score
    pred['status_score'] = status_score
    pred['intent'] = INTENT_DICT[intent_pred]

    return pred

  def compute_loss(self, batch, return_output=False):
    # first compute our regular forced decoding loss
    token_loss, model_output = super().compute_loss(batch, return_output=True)

    if not batch['is_outputact'].any():
      loss = token_loss
    else:
      loss = token_loss

      # only caculate loss at the end of each dialog (is_outputact == True)
      is_outputact = batch['is_outputact']
      bsz = is_outputact.long().sum().item()
      self.metrics['bsz'] += bsz

      encoder_states = model_output[2]
      has_res = batch['has_res'][is_outputact]

      # intent accuracy and loss
      intent_out = encoder_states[3][is_outputact]
      intent_labels = batch['intent_ids'][is_outputact]

      intent_loss = th.nn.functional.cross_entropy(
          intent_out, intent_labels, reduction='mean')
      loss += intent_loss * self.wei_intent
      _, intent_pred = intent_out.max(1)
      intent_acc = (intent_pred == intent_labels).float().sum().item()
      self.metrics['intent_acc'] += intent_acc
      self.metrics['intent_loss'] += bsz * intent_loss.item()

      # name accuracy and loss
      name_out = encoder_states[4][is_outputact]
      name_labels = batch['name_vec'][is_outputact]

      name_loss = th.nn.functional.cross_entropy(
          name_out.view(bsz * self.name_vec_len, -1),
          name_labels.view(-1),
          reduction='mean')
      loss += name_loss * self.wei_name
      _, name_pred = name_out.max(-1)
      name_acc = (name_pred == name_labels).all(dim=1).float().sum().item()
      self.metrics['name_acc'] += name_acc
      self.metrics['name_loss'] += bsz * name_loss.item()

      # flight loss
      need_flight_loss = (intent_labels == INTENT_DICT['book']) | \
          ((intent_labels == INTENT_DICT['change']) & has_res)
      flight_labels = batch['flight_ids'][is_outputact]
      ticket_attn = encoder_states[2][is_outputact]
      _, flight_pred = ticket_attn.max(-1)
      if need_flight_loss.any():
        flight_loss = th.nn.functional.cross_entropy(
            ticket_attn[need_flight_loss],
            flight_labels[need_flight_loss],
            reduction='mean')
        loss += flight_loss * self.wei_flight
        # the log one is not the exact flight loss
        self.metrics['flight_loss'] += bsz * flight_loss.item()

      # get actual status
      # import ipdb; ipdb.set_trace()
      status_labels = batch['status_ids'][is_outputact]
      pred_status_flight = [
          intent_to_status(f.item(), r.item(), i.item())
          for f, r, i in zip(flight_pred, has_res, intent_pred)
      ]
      pred_status_flight = th.LongTensor(pred_status_flight).to(intent_pred)
      flight_acc = (
          pred_status_flight[:, 1] == flight_labels).float().sum().item()
      status_acc = (
          pred_status_flight[:, 0] == status_labels).float().sum().item()
      self.metrics['flight_acc'] += flight_acc
      self.metrics['status_acc'] += status_acc
      if ((intent_pred == intent_labels) &
          (pred_status_flight[:, 1] == flight_labels) &
          (pred_status_flight[:, 0] != status_labels)).any():
        print(((intent_pred == intent_labels) &
               (pred_status_flight[:, 1] == flight_labels) &
               (pred_status_flight[:, 0] != status_labels)))
        import ipdb
        ipdb.set_trace()

    if return_output:
      return (loss, model_output)
    else:
      return loss

  def reset_metrics(self):
    super().reset_metrics()
    self.metrics['bsz'] = 0.0
    self.metrics['intent_acc'] = 0.0
    self.metrics['flight_acc'] = 0.0
    self.metrics['name_acc'] = 0.0
    self.metrics['status_acc'] = 0.0
    self.metrics['intent_loss'] = 0.0
    self.metrics['flight_loss'] = 0.0
    self.metrics['name_loss'] = 0.0

  def report(self):
    r = super().report()
    bsz = max(self.metrics['bsz'], 1)
    for k in [
        'intent_acc', 'flight_acc', 'name_acc', 'status_acc', 'intent_loss',
        'flight_loss', 'name_loss'
    ]:
      r[k] = round_sigfigs(self.metrics[k] / bsz, 4)
    return r

  def observe(self, obs):
    return super().observe(obs)

  def self_observe(self, obs):
    return super().self_observe(obs)

  def batchify(self, obs_batch):
    """
        Air custom batchify, which passes along the intent.

        Following the docstring of TorchAgent.batchify, it calls super, then
        uses an extended version of the torch_agent.Batch namedtuple.

        The purpose of extending the info is to keep track of some custom
        metrics.
        """
    #import ipdb; ipdb.set_trace()
    batch = super().batchify(obs_batch)
    reordered_observations = [obs_batch[i] for i in batch.valid_indices]
    is_training = 'labels' in reordered_observations[0]

    # first parse and compile all the knowledge together
    all_knowledges = []  # list-of-lists knowledge items for each observation
    all_reservations = []  # list of reservations for each observation
    knowledge_counts = []  # how much knowledge each observation gets
    for obs in reordered_observations:
      all_knowledges.append(obs['tickets'])
      all_reservations.append(obs['reservation'])
      knowledge_counts.append(len(obs['tickets']))

    # now we want to actually pack this into a tensor, along with the mask
    N = len(reordered_observations)
    K = max(knowledge_counts)
    # round out the array so everything is equally sized
    for i in range(N):
      all_knowledges[i] += [''] * (K - knowledge_counts[i])
    flattened_knowledge = list(chain(*all_knowledges))

    knowledge_vec = [
        self._vectorize_text(
            # the beginning of the sentence is more useful
            k,
            truncate=self.truncate,
            add_end=True,
            truncate_left=False,
        ) for k in flattened_knowledge
    ]
    knowledge_vec, _ = padded_tensor(knowledge_vec, self.NULL_IDX,
                                     self.use_cuda)
    #import ipdb; ipdb.set_trace() # check if the following line is valid
    T = knowledge_vec.size(-1)
    knowledge_vec = knowledge_vec.view(N, K, T)

    # knowledge mask is a N x K tensor saying which items we're allowed to
    # attend over
    ck_mask = th.zeros(N, K, dtype=th.uint8)
    for i, klen in enumerate(knowledge_counts):
      ck_mask[i, :klen] = 1
    ck_mask = ck_mask != 0  # for pytorch 1.0/1.2 uint8/bool compatibility
    # and the correct labels

    # gather reservation vector
    reservation_vec = [
        self._vectorize_text(
            # the beginning of the sentence is more useful
            r,
            truncate=self.truncate,
            add_end=True,
            truncate_left=False,
        ) for r in all_reservations
    ]
    reservation_vec, _ = padded_tensor(reservation_vec, self.NULL_IDX,
                                       self.use_cuda)

    # gather action
    intent_ids = []
    status_ids = []
    flight_ids = []
    is_outputact = []
    name_vec = []
    has_res = []
    for obs in reordered_observations:
      intent_ids.append(INTENT_DICT[obs['action_intent']])
      if obs['action_status'] == 'change':
        import ipdb
        ipdb.set_trace()
      status_ids.append(STATUS_DICT[obs['action_status']])
      if len(obs['action_flight']) == 0:
        flight_ids.append(0)
      else:
        flight_ids.append(obs['action_flight'][0] - 1000 + 1)
      name_vec.append(
          self._vectorize_text(
              obs['action_name'],
              truncate=self.name_vec_len,
              truncate_left=False,
          ))
      is_outputact.append(obs['episode_done'])
      has_res.append(False if obs['reservation'].lower()
                     .startswith('reservation none') else True)
    intent_ids = th.LongTensor(intent_ids)
    status_ids = th.LongTensor(status_ids)
    flight_ids = th.LongTensor(flight_ids)
    is_outputact = th.BoolTensor(is_outputact)
    has_res = th.BoolTensor(has_res)
    name_vec, name_vec_len = padded_tensor(
        name_vec, self.NULL_IDX, self.use_cuda, max_len=self.name_vec_len)
    if max(name_vec_len) > self.name_vec_len:
      print(f"OOD Name! {name_vec.shape[1]} increase name-vec-len")
      import ipdb
      ipdb.set_trace()

    # cuda
    if self.use_cuda:
      knowledge_vec = knowledge_vec.cuda()
      reservation_vec = reservation_vec.cuda()
      ck_mask = ck_mask.cuda()
      intent_ids = intent_ids.cuda()
      status_ids = status_ids.cuda()
      flight_ids = flight_ids.cuda()
      name_vec = name_vec.cuda()
      is_outputact = is_outputact.cuda()
      has_res = has_res.cuda()

    batch['know_vec'] = knowledge_vec
    batch['ck_mask'] = ck_mask
    batch['knowledge'] = np.array(flattened_knowledge).reshape(N, K)
    batch['res_vec'] = reservation_vec
    batch['intent_ids'] = intent_ids
    batch['status_ids'] = status_ids
    batch['flight_ids'] = flight_ids
    batch['name_vec'] = name_vec
    batch['is_outputact'] = is_outputact
    batch['has_res'] = has_res
    # import ipdb; ipdb.set_trace()
    return batch

  @classmethod
  def add_cmdline_args(cls, argparser):
    super(AgentAgent, cls).add_cmdline_args(argparser)
    group = argparser.add_argument_group('Agent Agent')
    group.add_argument(
        '--name-vec-len', type=int, default=10, help='max len of name vector')
    group.add_argument(
        '--wei-intent', type=float, default=1, help='weight for intent loss')
    group.add_argument(
        '--wei-flight', type=float, default=1, help='weight for flight loss')
    group.add_argument(
        '--wei-name', type=float, default=1, help='weight for name loss')

  def _model_input(self, batch):
    return (
        batch.text_vec,
        batch.know_vec,
        batch.ck_mask,
        batch.res_vec,
    )

  def build_model(self):
    self.model = EndToEndModel(self.opt, self.dict, agenttype='agent')
    if self.opt['embedding_type'] != 'random':
      self._copy_embeddings(self.model.embeddings.weight,
                            self.opt['embedding_type'])
    if self.use_cuda:
      self.model = self.model.cuda()
    return self.model


class CustomerAgent(_GenericAirAgent):

  def __init__(self, opt, shared=None):
    super().__init__(opt, shared)
    self._vectorize_text = lru_cache(int(2**20))(self._vectorize_text)
    self.id = 'Customer'

  def _dummy_batch(self, bsz, maxlen):
    batch = super()._dummy_batch(bsz, maxlen)
    batch['know_vec'] = th.zeros(bsz, 1, 2).long().cuda()
    # bool/uint8 backwards for pytorch 1.0/1.2 compatibility
    ck_mask = (th.ones(bsz, 1, dtype=th.uint8) != 0).cuda()
    batch['ck_mask'] = ck_mask
    return batch

  def compute_loss(self, batch, return_output=False):
    # first compute our regular forced decoding loss
    token_loss, model_output = super().compute_loss(batch, return_output=True)
    loss = token_loss
    if return_output:
      return (loss, model_output)
    else:
      return loss

  def reset_metrics(self):
    super().reset_metrics()

  def report(self):
    r = super().report()
    return r

  def observe(self, obs):
    return super().observe(obs)

  def self_observe(self, obs):
    return super().self_observe(obs)

  def batchify(self, obs_batch):
    """
        Air custom batchify, which passes along the intent.

        Following the docstring of TorchAgent.batchify, it calls super, then
        uses an extended version of the torch_agent.Batch namedtuple.

        The purpose of extending the info is to keep track of some custom
        metrics.
        """
    # import ipdb; ipdb.set_trace()
    batch = super().batchify(obs_batch)
    reordered_observations = [obs_batch[i] for i in batch.valid_indices]
    is_training = 'labels' in reordered_observations[0]

    # first parse and compile all the knowledge together
    all_knowledges = []  # list-of-lists knowledge items for each observation
    for obs in reordered_observations:
      obs_know = [obs.get('intent')]
      all_knowledges.append(obs_know)

    # now we want to actually pack this into a tensor, along with the mask
    N = len(reordered_observations)
    K = 1

    flattened_knowledge = list(chain(*all_knowledges))
    knowledge_vec = [
        self._vectorize_text(
            # the beginning of the sentence is more useful
            k,
            truncate=self.truncate,
            add_end=True,
            truncate_left=False,
        ) for k in flattened_knowledge
    ]
    knowledge_vec, _ = padded_tensor(knowledge_vec, self.NULL_IDX,
                                     self.use_cuda)
    T = knowledge_vec.size(-1)
    knowledge_vec = knowledge_vec.view(N, K, T)

    # knowledge mask is a N x K tensor saying which items we're allowed to
    # attend over
    ck_mask = th.ones(N, K, dtype=th.uint8)
    ck_mask = ck_mask != 0  # for pytorch 1.0/1.2 uint8/bool compatibility

    if self.use_cuda:
      knowledge_vec = knowledge_vec.cuda()
      ck_mask = ck_mask.cuda()

    batch['know_vec'] = knowledge_vec
    batch['ck_mask'] = ck_mask
    batch['knowledge'] = np.array(flattened_knowledge).reshape(N, K)
    return batch

  @classmethod
  def add_cmdline_args(cls, argparser):
    super(CustomerAgent, cls).add_cmdline_args(argparser)
    group = argparser.add_argument_group('Customer Agent')

  def _model_input(self, batch):
    return (
        batch.text_vec,
        batch.know_vec,
        batch.ck_mask,
    )

  def build_model(self):
    self.model = EndToEndModel(self.opt, self.dict, agenttype='customer')
    if self.opt['embedding_type'] != 'random':
      self._copy_embeddings(self.model.embeddings.weight,
                            self.opt['embedding_type'])
    if self.use_cuda:
      self.model = self.model.cuda()
    return self.model
