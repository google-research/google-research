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

from transformers.modeling_roberta import *
from .modeling_bert import BertPreTrainedModels
import torch
import torch.nn.functional as F

from .utils import REGFUNC


@add_start_docstrings("""2 RoBERTa model : a QNet and CNet. """,
                      ROBERTA_START_DOCSTRING)
class RobertaForAirOPE(BertPreTrainedModels):
  config_class = RobertaConfig
  base_model_prefix = 'roberta'
  base_model_prefixs = ['roberta_qnet', 'roberta_cnet']

  def __init__(self, config, args):
    # hack roberta token type embeddings
    config.type_vocab_size = 2
    super().__init__(config)

    self.roberta_qnet = RobertaModel(config)
    self.qnet_head = RobertaQHead(config, finalact=args.finalact_q)

    self.share_roberta = args.share_bert
    self.fix_roberta = args.fix_bert
    self.freeze_roberta = args.freeze_bert

    if self.share_roberta:
      self.base_model_prefixs = ['roberta_qnet']

    self.roberta_cnet = RobertaModel(config) if not self.share_roberta else None
    self.cnet_head = RobertaCHead(config, finalact=args.finalact_c)

    self.pos_padding_idx = config.pad_token_id

    self.gamma = args.gamma
    self._lamb = nn.Parameter(torch.ones(1) *
                              args.lambinit) if args.normalize_c else 0
    self.alphaR = args.alphaR
    self.alphaQ = args.alphaQ
    self.regfunQ = REGFUNC[args.regfunQ]
    self.alphaC = args.alphaC
    self.regfunC = REGFUNC[args.regfunC]
    self.alphaL = args.alphaL
    self.regfunL = REGFUNC[args.regfunL]

    # Cotraining Objective
    self.alphaAux = args.alphaAux
    self.detach_roberta_qc = False
    if self.alphaAux > 0:
      self.auxnet_head = RobertaQHead(config, finalact=args.finalact_aux)
      assert self.share_roberta
      if self.fix_roberta:
        self.fix_roberta = False
        self.detach_roberta_qc = True

    # Pseudo State
    self.max_turns = args.max_turns
    self._q_pseudo = nn.Parameter(torch.zeros(self.max_turns))
    self._c_pseudo = nn.Parameter(torch.ones(self.max_turns))

    self.normalize_obj_by_turns = args.normalize_obj_by_turns

    # LR scales
    self.lrscale_bert = args.lrscale_bert
    self.lrscale_lamb = args.lrscale_lamb
    self.scale_lamb = args.scale_lamb
    self.lrscale_c = args.lrscale_c
    self.lrscale_q = args.lrscale_q

    self.init_weights()

  @property
  def lamb(self):
    return grad_scale(self._lamb, self.lrscale_lamb) * self.scale_lamb

  @property
  def q_pseudo(self):
    return grad_scale(self._q_pseudo, self.lrscale_q)

  @property
  def c_pseudo(self):
    return grad_reverse(self._c_pseudo, self.lrscale_c)

  def init_weights(self):
    super().init_weights()
    # hack roberta token type embeddings
    self.roberta_qnet.embeddings.token_type_embeddings.weight.data.zero_()

  def hack_pretrained_state_dict(self, state_dict):
    for key in state_dict.keys():
      if key.endswith('token_type_embeddings.weight'):
        _state = state_dict[key]
        if _state.shape[0] < self.config.type_vocab_size:
          assert _state.shape[0] == 1
          state_dict[key] = torch.cat(
              [_state] * self.config.type_vocab_size, dim=0)

    return state_dict

  def extract_features(self, *args, **kargs):
    if self.freeze_roberta:
      self.roberta_qnet.eval()
      if not self.share_roberta:
        self.roberta_cnet.eval()
    if self.fix_roberta:
      with torch.no_grad():
        qnet_outputs = self.roberta_qnet(*args, **kargs)
        if not self.share_roberta:
          cnet_outputs = self.roberta_cnet(*args, **kargs)
        else:
          cnet_outputs = qnet_outputs
    else:
      qnet_outputs = self.roberta_qnet(*args, **kargs)
      if not self.share_roberta:
        cnet_outputs = self.roberta_cnet(*args, **kargs)
      else:
        cnet_outputs = qnet_outputs
    return qnet_outputs, cnet_outputs

  @add_start_docstrings_to_callable(
      ROBERTA_INPUTS_DOCSTRING.format('(batch_size, sequence_length)'))
  def forward(self,
              input_ids,
              attention_mask,
              token_type_ids,
              position_ids,
              reward,
              true_conv_end,
              ref_c_end_ids,
              ref_a_end_ids,
              gen_a_end_ids,
              head_mask=None,
              inputs_embeds=None,
              output_attentions=None,
              detach_cnet=False,
              detach_qnet=False,
              **kwargs):
    assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

    if isinstance(reward, dict):
      # reward is used for training; other entries are only used in ope estimation
      reward_dict = reward
      reward = reward['reward']
    else:
      reward_dict = None

    # import ipdb;ipdb.set_trace()
    # Roberta inputid hack
    pos_ids_mask = (position_ids > 0)
    pos_ids_mask[:, 0] = True
    position_ids = (position_ids +
                    1) * pos_ids_mask.int() + self.pos_padding_idx

    qnet_outputs, cnet_outputs = self.extract_features(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
    )

    if self.detach_roberta_qc:
      tmp = qnet_outputs[0].detach()
    else:
      tmp = grad_scale(qnet_outputs[0], self.lrscale_bert)
    q_values = self.qnet_head(tmp)
    q_values = grad_scale(q_values, self.lrscale_q)
    if detach_qnet:
      q_values = q_values.detach()

    if self.detach_roberta_qc:
      tmp = cnet_outputs[0].detach()
    else:
      tmp = grad_scale(cnet_outputs[0], self.lrscale_bert)
    c_values = self.cnet_head(tmp)
    c_values = grad_reverse(c_values, self.lrscale_c)
    if detach_cnet:
      c_values = c_values.detach()

    if self.alphaAux > 0:
      tmp = grad_scale(qnet_outputs[0], self.lrscale_bert)
      aux_q_values = self.auxnet_head(tmp)

    outputs = (q_values, c_values, qnet_outputs, cnet_outputs
              )  # Add hidden states and attention if they are here

    # handle pseudo state
    q_gen_s0a0 = self.q_pseudo[0]
    q_gen_s0a0 = q_gen_s0a0 * (1 - self.gamma)

    # extrate c,q values
    batch_size = ref_a_end_ids.shape[0]
    turns = ref_a_end_ids.shape[1]
    assert turns + 1 < self.max_turns, 'Turns: {} self.max_turns: {}'.format(
        turns + 1, self.max_turns)
    ref_a_end_ids = torch.cat([
        torch.ones(batch_size, 1).to(ref_a_end_ids) * -1,
        ref_a_end_ids,
        torch.ones(batch_size, self.max_turns - turns - 1).to(ref_a_end_ids) *
        -1,
    ],
                              dim=1)
    gen_a_end_ids = torch.cat([
        torch.ones(batch_size, 1).to(gen_a_end_ids) * -1,
        gen_a_end_ids,
        torch.ones(batch_size, self.max_turns - turns - 1).to(gen_a_end_ids) *
        -1,
    ],
                              dim=1)
    ref_a_mask = (ref_a_end_ids == -1)
    gen_a_mask = (gen_a_end_ids == -1)
    c_ref_values =  self.c_pseudo.expand(batch_size, -1).masked_fill(~ref_a_mask,0) + \
        c_values.gather(dim=1,index=ref_a_end_ids.masked_fill(ref_a_mask,0)).masked_fill(ref_a_mask,0)
    q_ref_values =  self.q_pseudo.expand(batch_size, -1).masked_fill(~ref_a_mask,0) + \
        q_values.gather(dim=1,index=ref_a_end_ids.masked_fill(ref_a_mask,0)).masked_fill(ref_a_mask,0)
    q_gen_values =  self.q_pseudo.expand(batch_size, -1).masked_fill(~gen_a_mask,0) + \
        q_values.gather(dim=1,index=gen_a_end_ids.masked_fill(gen_a_mask,0)).masked_fill(gen_a_mask,0)
    # Shift q_gen_values
    q_gen_values = self.gamma * torch.cat(
        [q_gen_values[:, 1:], q_gen_values[:, 0:1]], dim=1)
    normalize_factor = self.max_turns if self.normalize_obj_by_turns else 1

    c_ref_term_values = c_values.gather(
        dim=1, index=true_conv_end.unsqueeze(-1)).squeeze()
    q_ref_term_values = q_values.gather(
        dim=1, index=true_conv_end.unsqueeze(-1)).squeeze()
    # import ipdb; ipdb.set_trace()
    # print(q_values.shape, c_values.shape)

    # Gather Cotraining aux_q_values
    if self.alphaAux > 0:
      aux_q_ref_values = aux_q_values.gather(
          dim=1, index=ref_a_end_ids.masked_fill(ref_a_mask, 0))
      reg_aux = ((reward[:, None] - aux_q_ref_values)**2).masked_fill(
          ref_a_mask, 0).sum(dim=-1) / normalize_factor
    else:
      reg_aux = torch.zeros(1).to(q_values)

    reg_q = self.regfunQ(q_ref_values).sum(dim=-1) / normalize_factor
    reg_c = self.regfunC(c_ref_values).sum(dim=-1) / normalize_factor
    reg_c_normal = (1 - c_ref_values).sum(dim=-1) / normalize_factor
    reg_l = self.regfunL(self.lamb)

    td_error = ((q_gen_values - q_ref_values).sum(dim=-1) +
                self.alphaR * reward) / normalize_factor
    corrected_td_error = ((c_ref_values*(q_gen_values - q_ref_values)).sum(dim=-1) + \
                          self.alphaR*c_ref_term_values*reward) / normalize_factor

    loss = q_gen_s0a0 + corrected_td_error \
        + self.alphaQ * reg_q - self.alphaC * reg_c + self.lamb * reg_c_normal + self.alphaL * reg_l \
        + self.alphaAux * reg_aux
    #loss = - self.alphaC * reg_c + self.lamb * reg_c_normal + self.alphaL * reg_l
    if torch.isnan(loss).any():
      import ipdb
      ipdb.set_trace()

    loggings = {
        'est_reward_qfunc': (q_gen_s0a0 + self.lamb)*normalize_factor,
        'est_reward_qfunc2': self.q_pseudo[0],
        'est_reward_dual': c_ref_term_values*reward,
        'est_reward_dual2': c_ref_term_values*reward/(2 * self.alphaL * self.lamb + 1 + 1e-5 ),
        'est_reward_lagr':  (q_gen_s0a0 + corrected_td_error + self.lamb * reg_c_normal)*normalize_factor \
                            + (1-self.alphaR)*c_ref_term_values*reward,
        'td_error': td_error,
        'corrected_td_error': corrected_td_error,
        'reg_q': reg_q,
        'reg_c': reg_c,
        'reg_c_normal': reg_c_normal,
        'reg_aux': reg_aux,
        'lambda': self.lamb,
        'q_gen_s0a0': q_gen_s0a0,
        'c_ref_values': c_ref_values,
        'q_ref_values': q_ref_values,
        'q_gen_values': q_gen_values,
        'c_ref_term_values': c_ref_term_values,
        'q_ref_term_values': q_ref_term_values,
    }
    if reward_dict is not None:
      for k, v in reward_dict.items():
        loggings['est_reward_dual_' + k] = c_ref_term_values * v

    outputs = (
        loss,
        loggings,
    ) + outputs

    return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class RobertaQHead(nn.Module):
  """Roberta Head for masked language modeling."""

  def __init__(self, config, finalact='sigmoid'):
    super().__init__()

    if finalact.startswith('linear-'):
      self.no_first = True
      finalact = finalact.replace('linear-', '')
    else:
      self.no_first = False
      self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    self.dense2 = nn.Linear(config.hidden_size, 1, bias=True)

    if finalact == 'sigmoid':
      self.finalact = nn.Sigmoid()
    elif finalact == 'no':
      self.finalact = lambda x: x

  def forward(self, x, **kwargs):
    if not self.no_first:
      x = self.dense(x)
      x = F.gelu(x)

    x = self.dense2(x).squeeze(-1)
    x = self.finalact(x)

    return x


class GradientScale(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x, scale):
    ctx.scale = scale
    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    scale = ctx.scale
    return scale * grad_output, None


def grad_reverse(x, scale=1.0):
  return GradientScale.apply(x, -scale)


def grad_scale(x, scale=1.0):
  return GradientScale.apply(x, scale)


class RobertaCHead(nn.Module):
  """Roberta Head for masked language modeling."""

  def __init__(self, config, finalact='square'):
    super().__init__()
    if finalact.startswith('linear-'):
      self.no_first = True
      finalact = finalact.replace('linear-', '')
    else:
      self.no_first = False
      self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    self.dense2 = nn.Linear(config.hidden_size, 1, bias=True)
    if finalact == 'square':
      self.finalact = lambda x: x**2
    elif finalact == 'square_bias1':
      self.finalact = lambda x: (x + 1)**2
    elif finalact == 'softplus':
      self.finalact = nn.Softplus()
    elif finalact == 'no':
      self.finalact = lambda x: x

  def forward(self, x, **kwargs):
    if not self.no_first:
      x = self.dense(x)
      x = F.gelu(x)

    x = self.dense2(x).squeeze(-1)
    x = self.finalact(x)

    return x
