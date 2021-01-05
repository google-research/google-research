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

# python3
"""Defines embedders for various cache objects."""

import abc
import torch
from torch import nn


def from_config(config):
  """Creates an embedder specified by the config.

  Args:
    config (cfg.Config): specifies embedder type and constructor args.

  Returns:
    Embedder: embedder specified by the config.
  """
  embedder_type = config.get("type")
  if embedder_type == "byte":
    return ByteEmbedder(config.get("bytes_per_entry"), config.get("embed_dim"))
  elif embedder_type == "dynamic-vocab":
    return DynamicVocabEmbedder(
        config.get("embed_dim"), config.get("max_vocab_size"))
  elif embedder_type == "positional":
    return PositionalEmbedder(config.get("embed_dim"))
  else:
    raise ValueError("{} not a supported embedding type.".format(embedder_type))


class Embedder(nn.Module):
  """Embeds a batch of objects into an embedding space.

  Subclasses of Embedder should register with the from_config method.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, embed_dim):
    """Sets the output embedding dimension to be embed_dim.

    Args:
      embed_dim (int): dimension of output of forward call.
    """
    super(Embedder, self).__init__()
    self._embed_dim = embed_dim

  @property
  def embed_dim(self):
    return self._embed_dim


class ByteEmbedder(Embedder):
  """Embeds each byte and concatenates."""

  def __init__(self, bytes_per_entry, embed_dim):
    """Embeds entries that have bytes_per_entry many bytes.

    Args:
      bytes_per_entry (int): number of bytes per input.
      embed_dim (int): see parent class.
    """
    super(ByteEmbedder, self).__init__(embed_dim)

    if embed_dim % bytes_per_entry != 0:
      raise ValueError(
          "Embed dim ({}) must be an even multiple of bytes per entry ({})"
          .format(embed_dim, bytes_per_entry))

    embed_dim_per_byte = embed_dim // bytes_per_entry
    # 256 possible byte values
    self._byte_embedding = nn.Embedding(256, embed_dim_per_byte)
    self._bytes_per_entry = bytes_per_entry
    self._final_layer = nn.Linear(embed_dim, embed_dim)

  def forward(self, ints):
    """Returns embeddings for each int interpretted as a byte array.

    Args:
      ints (list[int]): batch of inputs of length batch_size.

    Returns:
      embeddings (torch.FloatTensor): batch of embeddings of shape
        (batch_size, embed_dim). Each int is interpretted as bytes_per_entry
        bytes and each byte is embedded separately.
    """
    def int_to_byte_tensor(ints, num_bytes):
      """Converts ints to tensor of shape (num_bytes).

      Args:
        ints (list[int]): ints to convert.
        num_bytes (int): number of bytes to convert to.

      Returns:
        byte_tensor (torch.LongTensor): shape (len(ints), num_bytes).
          byte_tensor[i][j] = value of jth byte of ints[i].
      """
      # Byte order doesn't matter as long as it's consistent.
      return torch.tensor(
          [int(x).to_bytes(num_bytes, byteorder="big") for x in ints]).long()

    # (batch_size, bytes_per_entry, embed_dim_per_byte)
    byte_tensors = int_to_byte_tensor(ints, self._bytes_per_entry)
    byte_embeddings = self._byte_embedding(byte_tensors)
    return self._final_layer(byte_embeddings.view(-1, self.embed_dim))


class DynamicVocabEmbedder(Embedder):
  """Dynamically constructs a vocab, assigning embeddings to new inputs.

  After max_vocab_size unique inputs are observed, all new inputs are assigned
  to a UNK embedding.
  """

  def __init__(self, embed_dim, max_vocab_size):
    super().__init__(embed_dim)

    self._max_vocab_size = max_vocab_size
    self._input_to_index = {}
    # Reserve index 0 for UNK
    self._vocab_size = 1

    # Override default initialization of embeddings with Xavier
    weight = torch.zeros(max_vocab_size, embed_dim)
    nn.init.xavier_uniform_(weight)
    self._embedding = nn.Embedding(max_vocab_size, embed_dim, _weight=weight)

  def forward(self, inputs):
    """Returns embeddings for each int interpretted as a byte array.

    Args:
      inputs (list[Object]): batch of hashable inputs of length batch_size.

    Returns:
      embeddings (torch.FloatTensor): batch of embeddings of shape
        (batch_size, embed_dim).
    """
    def input_to_index(inp):
      if (inp not in self._input_to_index and
          self._max_vocab_size > self._vocab_size):
        self._input_to_index[inp] = self._vocab_size
        self._vocab_size += 1
      # Return index 0 (UNK) if vocab is full and inp is not in vocab
      return self._input_to_index.get(inp, 0)

    indices = torch.tensor([input_to_index(inp) for inp in inputs]).long()
    return self._embedding(indices)

  def state_dict(self, destination=None, prefix="", keep_vars=False):
    state_dict = super().state_dict(destination, prefix, keep_vars)
    state_dict[prefix + "vocab_size"] = self._vocab_size
    state_dict[prefix + "input_to_index"] = self._input_to_index
    return state_dict

  def _load_from_state_dict(self, state_dict, prefix, strict, missing_keys,
                            unexpected_keys, error_msgs):
    self._vocab_size = state_dict.pop(prefix + "vocab_size")
    self._input_to_index = state_dict.pop(prefix + "input_to_index")
    super()._load_from_state_dict(
        state_dict, prefix, strict, missing_keys, unexpected_keys, error_msgs)


class PositionalEmbedder(Embedder):
  """Takes position index and returns a simple fixed embedding."""

  def forward(self, position_indices):
    """Returns a fixed embedding for each input index.

    Embeds positions according to Vaswani, et. al., 2017:
      embed_{2i} = sin(pos / 10000^(2i / embed_dim))
      embed_{2i + 1} = cos(pos / 10000^(2i / embed_dim))

    Args:
      position_indices (list[int]): batch of positions of length batch_size

    Returns:
      embeddings (torch.FloatTensor): of shape (batch_size, embed_dim)
    """
    batch_size = len(position_indices)

    # i's in above equation
    embed_indices = torch.arange(self.embed_dim).expand(batch_size, -1).float()
    position_tensor = torch.tensor(position_indices).unsqueeze(-1).float()
    embedding = position_tensor / 10000. ** (2 * embed_indices / self.embed_dim)
    embedding = torch.where(
        embed_indices % 2 == 0, torch.sin(embedding), torch.cos(embedding))
    return embedding
