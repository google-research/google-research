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


from typing import Dict

import numpy as np
import torch as th
import torch.nn as nn

from parlai.utils.torch import neginf
from parlai.agents.transformer.modules import TransformerGeneratorModel, TransformerEncoder


def universal_sentence_embedding(sentences, mask, sqrt=True):
  """
    Perform Universal Sentence Encoder averaging
    (https://arxiv.org/abs/1803.11175).

    This is really just sum / sqrt(len).

    :param Tensor sentences: an N x T x D of Transformer outputs. Note this is
        the exact output of TransformerEncoder, but has the time axis first
    :param ByteTensor: an N x T binary matrix of paddings

    :return: an N x D matrix of sentence embeddings
    :rtype Tensor:
    """
  # need to mask out the padded chars
  sentence_sums = th.bmm(
      sentences.permute(0, 2, 1),
      mask.float().unsqueeze(-1)).squeeze(-1)
  divisor = mask.sum(dim=1).view(-1, 1).float()
  if sqrt:
    divisor = divisor.sqrt()
  sentence_sums /= divisor
  return sentence_sums


class EndToEndModel(TransformerGeneratorModel):

  def __init__(self, opt, dictionary, agenttype):
    super().__init__(opt, dictionary)
    self.encoder = ContextKnowledgeEncoder(self.encoder, opt, dictionary,
                                           agenttype)
    self.decoder = ContextKnowledgeDecoder(self.decoder, agenttype)
    self.agenttype = agenttype

  def reorder_encoder_states(self, encoder_out, indices):
    # ck_attn is used for ticket classification
    enc, mask, ck_attn, intent_out, name_out = encoder_out
    if not th.is_tensor(indices):
      indices = th.LongTensor(indices).to(enc.device)
    enc = th.index_select(enc, 0, indices)
    mask = th.index_select(mask, 0, indices)
    if self.agenttype == 'agent':
      intent_out = th.index_select(intent_out, 0, indices)
      name_out = th.index_select(name_out, 0, indices)
      ck_attn = th.index_select(ck_attn, 0, indices)
    else:
      intent_out = None
      ck_attn = None
      name_out = None
    return enc, mask, ck_attn, intent_out, name_out

  def reorder_decoder_incremental_state(self, incremental_state,
                                        inds):
    """
        Reorder the decoder incremental state.

        See ``TorchGeneratorModel.reorder_decoder_incremental_state`` for a
        description.

        Here, incremental_state is a dict whose keys are layer indices and whose
        values
        are dicts containing the incremental state for that layer.
        """
    return {
        idx: layer.reorder_incremental_state(incremental_state[idx], inds)
        for idx, layer in enumerate(self.decoder.transformer.layers)
    }


class ClassificationHead(nn.Module):

  def __init__(self, dim, out=3):
    """
        3 classes: book, cancel, change
        """
    super().__init__()
    self.linear = nn.Linear(dim, dim)
    self.attn_wei = nn.Linear(dim, 1)
    self.softmax = nn.Softmax(dim=1)
    self.act = nn.Tanh()
    self.final = nn.Linear(dim, out)

  def forward(self, x, mask):
    x = self.linear(x)
    x = self.act(x)

    attn = self.attn_wei(x).squeeze(-1)
    attn.masked_fill_(~mask, neginf(x.dtype))
    attn = self.softmax(attn)

    x = th.einsum('btd,bt->bd', x, attn)

    x = self.final(x)
    return x


class MultiTokenClassificationHead(nn.Module):

  def __init__(self, dim, embeddings, out=10):
    super().__init__()
    self.linear = nn.Linear(dim, out * dim)
    self.attn_wei = nn.Linear(dim, 1)
    self.act = nn.Tanh()
    self.softmax = nn.Softmax(dim=1)

    self.proj = nn.Linear(dim, dim)
    self.embeddings = embeddings.weight
    self.out = out

  def forward(self, x, mask):
    # import ipdb; ipdb.set_trace()
    # x: N x T x D
    N, T, D = x.shape
    x = self.linear(x).view(N, T, self.out, D)
    x = self.act(x)

    attn = self.attn_wei(x).squeeze(-1)
    attn.masked_fill_(~mask[:, :, None], neginf(x.dtype))
    attn = self.softmax(attn)

    x = th.einsum('btod,bto->bod', x, attn)
    x = self.proj(x)

    x = th.einsum('bod,vd->bov', x, self.embeddings)
    return x


class ContextKnowledgeEncoder(nn.Module):
  """
    Knowledge here can be customer intent or tickets+reservations
    """

  def __init__(self, transformer, opt, dictionary, agenttype):
    super().__init__()
    # The transformer takes care of most of the work, but other modules
    # expect us to have an embeddings available
    self.embeddings = transformer.embeddings
    self.embed_dim = transformer.embeddings.embedding_dim
    self.transformer = transformer
    self.knowledge_transformer = TransformerEncoder(
        embedding=self.embeddings,
        n_heads=opt['n_heads'],
        n_layers=opt['n_layers_knowledge'],
        embedding_size=opt['embedding_size'],
        ffn_size=opt['ffn_size'],
        vocabulary_size=len(dictionary),
        padding_idx=transformer.padding_idx,
        learn_positional_embeddings=opt['learn_positional_embeddings'],
        embeddings_scale=opt['embeddings_scale'],
        reduction_type=transformer.reduction_type,
        n_positions=transformer.n_positions,
        activation=opt['activation'],
        variant=opt['variant'],
        output_scaling=opt['output_scaling'],
    )
    self.agenttype = agenttype
    if self.agenttype == 'agent':
      self.intent_head = ClassificationHead(opt['embedding_size'])
      self.name_head = MultiTokenClassificationHead(opt['embedding_size'],
                                                    self.embeddings,
                                                    opt.get('name_vec_len'))
      self.reservation_transformer = TransformerEncoder(
          embedding=self.embeddings,
          n_heads=opt['n_heads'],
          n_layers=opt['n_layers_knowledge'],
          embedding_size=opt['embedding_size'],
          ffn_size=opt['ffn_size'],
          vocabulary_size=len(dictionary),
          padding_idx=transformer.padding_idx,
          learn_positional_embeddings=opt['learn_positional_embeddings'],
          embeddings_scale=opt['embeddings_scale'],
          reduction_type=transformer.reduction_type,
          n_positions=transformer.n_positions,
          activation=opt['activation'],
          variant=opt['variant'],
          output_scaling=opt['output_scaling'],
      )
      self.know_use_project = nn.Linear(opt['embedding_size'],
                                        opt['embedding_size'])

  def forward(self, src_tokens, know_tokens, ck_mask, res_tokens=None):
    # encode the context, pretty basic
    context_encoded, context_mask = self.transformer(src_tokens)

    # make all the knowledge into a 2D matrix to encode
    # knowledge is intent for customer and tickets for agent
    N, K, Tk = know_tokens.size()
    know_flat = know_tokens.reshape(-1, Tk)
    know_encoded, know_mask = self.knowledge_transformer(know_flat)

    if self.agenttype == 'customer':
      ck_attn = None
      intent_out = None
      name_out = None
      cs_encoded = know_encoded
      cs_mask = know_mask
    elif self.agenttype == 'agent':
      # import ipdb; ipdb.set_trace()

      # compute our sentence embeddings for context and knowledge
      context_use = universal_sentence_embedding(context_encoded, context_mask)
      know_use = universal_sentence_embedding(know_encoded, know_mask)

      # remash it back into the shape we need
      know_use = know_use.reshape(N, K, self.embed_dim)
      # project before calculate attn
      know_use_proj = self.know_use_project(know_use)
      ck_attn = th.bmm(know_use_proj, context_use.unsqueeze(-1)).squeeze(-1)
      ck_attn /= np.sqrt(self.embed_dim)
      # fill with near -inf
      ck_attn.masked_fill_(~ck_mask, neginf(context_encoded.dtype))

      # Compute context knowledge attn prob
      ck_prob = nn.functional.softmax(ck_attn, dim=-1)

      _, cs_ids = ck_attn.max(1)

      # pick the true chosen sentence. remember that TransformerEncoder outputs
      #   (batch, time, embed)
      # but because know_encoded is a flattened, it's really
      #   (N * K, T, D)
      # We need to compute the offsets of the chosen_sentences
      cs_offsets = th.arange(N, device=cs_ids.device) * K + cs_ids
      cs_encoded = know_encoded[cs_offsets]
      # but padding is (N * K, T)
      cs_mask = know_mask[cs_offsets]

      # compute reservation embeddings
      res_encoded, res_mask = self.reservation_transformer(res_tokens)

      # finally, concatenate it all
      cs_encoded = th.cat([know_use, cs_encoded, res_encoded], dim=1)
      cs_mask = th.cat([ck_mask, cs_mask, res_mask], dim=1)

      # intent prediction
      intent_out = self.intent_head(context_encoded, context_mask)
      name_out = self.name_head(context_encoded, context_mask)

    # finally, concatenate it all
    full_enc = th.cat([cs_encoded, context_encoded], dim=1)
    full_mask = th.cat([cs_mask, context_mask], dim=1)

    # also return the knowledge selection mask for the loss
    return full_enc, full_mask, ck_attn, intent_out, name_out


class ContextKnowledgeDecoder(nn.Module):

  def __init__(self, transformer, agenttype):
    super().__init__()
    self.transformer = transformer
    self.agenttype = agenttype

  def forward(self, input, encoder_state, incr_state=None):
    # our CK Encoder returns an extra output which the Transformer decoder
    # doesn't expect (the knowledge selection mask). Just chop it off
    encoder_output, encoder_mask, _, _, _ = encoder_state
    return self.transformer(input, (encoder_output, encoder_mask), incr_state)
