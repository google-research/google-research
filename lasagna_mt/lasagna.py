# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Code for the Lasagna model for MT."""
import math

from fairseq import utils
from fairseq.models import (  # pylint: disable=g-multiple-import
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (  # pylint: disable=g-multiple-import
    AdaptiveSoftmax,
    DynamicConv,
    FairseqDropout,
    LayerNorm,
    LightweightConv,
    MultiheadAttention,
    PositionalEmbedding,
)
from fairseq.utils import safe_hasattr
import torch
from torch import nn
import torch.nn.functional as F


@register_model("lasagna")
class LasagnaModel(FairseqEncoderDecoderModel):
  """Allows to stacking different layers in different orders."""

  def __init__(self, encoder, decoder):  # pylint: disable = useless-parent-delegation
    super().__init__(encoder, decoder)

  @staticmethod
  def add_args(parser):
    """Add model-specific arguments to the parser."""
    parser.add_argument(
        "--dropout", type=float, metavar="D", help="dropout probability"
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        metavar="D",
        help="dropout probability for attention weights",
    )
    parser.add_argument(
        "--relu-dropout",
        type=float,
        metavar="D",
        help="dropout probability after ReLU in FFN",
    )
    parser.add_argument(
        "--input-dropout",
        type=float,
        metavar="D",
        help="dropout probability of the inputs",
    )
    parser.add_argument(
        "--encoder-embed-path",
        type=str,
        metavar="STR",
        help="path to pre-trained encoder embedding",
    )
    parser.add_argument(
        "--encoder-embed-dim",
        type=int,
        metavar="N",
        help="encoder embedding dimension",
    )
    parser.add_argument(
        "--encoder-ffn-embed-dim",
        type=int,
        metavar="N",
        help="encoder embedding dimension for FFN",
    )
    parser.add_argument(
        "--encoder-layers", type=int, metavar="N", help="num encoder layers"
    )
    parser.add_argument(
        "--encoder-normalize-before",
        action="store_true",
        help="apply layernorm before each encoder block",
    )
    parser.add_argument(
        "--encoder-learned-pos",
        action="store_true",
        help="use learned positional embeddings in the encoder",
    )
    parser.add_argument(
        "--decoder-embed-path",
        type=str,
        metavar="STR",
        help="path to pre-trained decoder embedding",
    )
    parser.add_argument(
        "--decoder-deep-embedding-layers",
        type=int,
        metavar="N",
        help="number of layers for the deep embedding",
        default=0,
    )
    parser.add_argument(
        "--deep-embedding-dropout",
        type=float,
        metavar="N",
        help="Dropout for the deep embedding",
        default=0.1,
    )
    parser.add_argument(
        "--decoder-embed-dim",
        type=int,
        metavar="N",
        help="decoder embedding dimension",
    )
    parser.add_argument(
        "--decoder-ffn-embed-dim",
        type=int,
        metavar="N",
        help="decoder embedding dimension for FFN",
    )
    parser.add_argument(
        "--decoder-layers", type=int, metavar="N", help="num decoder layers"
    )
    parser.add_argument(
        "--decoder-learned-pos",
        action="store_true",
        help="use learned positional embeddings in the decoder",
    )
    parser.add_argument(
        "--decoder-normalize-before",
        action="store_true",
        help="apply layernorm before each decoder block",
    )
    parser.add_argument(
        "--share-decoder-input-output-embed",
        action="store_true",
        help="share decoder input and output embeddings",
    )
    parser.add_argument(
        "--share-all-embeddings",
        action="store_true",
        help=(
            "share encoder, decoder and output embeddings"
            " (requires shared dictionary and embed dim)"
        ),
    )
    parser.add_argument(  # pylint: disable=expression-not-assigned
        "--adaptive-softmax-cutoff",
        metavar="EXPR",
        help=(
            "comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion"
        ),
    ),
    parser.add_argument(
        "--adaptive-softmax-dropout",
        type=float,
        metavar="D",
        help="sets adaptive softmax dropout for the tail projections",
    )

    parser.add_argument(
        "--encoder-kernel-size-list",
        type=lambda x: utils.eval_str_list(x, int),
        help='list of kernel size (default: "[3,7,15,31,31,31,31]")',
    )
    parser.add_argument(
        "--decoder-kernel-size-list",
        type=lambda x: utils.eval_str_list(x, int),
        help='list of kernel size (default: "[3,7,15,31,31,31]")',
    )
    parser.add_argument(
        "--encoder-layer-type-list",
        type=lambda x: utils.eval_str_list(x, str),
        help='Choose in "self", "conv", "gmlp"',
    )
    parser.add_argument(
        "--encoder-mlp-type-list",
        type=lambda x: utils.eval_str_list(x, str),
        help='"mlp", "no", "gmlp_l", "gmlp_d"',
    )
    parser.add_argument(
        "--decoder-layer-type-list",
        type=lambda x: utils.eval_str_list(x, str),
        help='"self", "conv", "gmlp"',
    )
    parser.add_argument(
        "--decoder-mlp-type-list",
        type=lambda x: utils.eval_str_list(x, str),
        help='"mlp", "no", "gmlp_l", "gmlp_d"',
    )
    parser.add_argument(
        "--encoder-conv-type-list",
        type=lambda x: utils.eval_str_list(x, str),
        help='Choose in "dynamic", "lightweight"',
    )
    parser.add_argument(
        "--decoder-conv-type-list",
        type=lambda x: utils.eval_str_list(x, str),
        help='Choose in "dynamic", "lightweight", "dynamic_experts"',
    )
    parser.add_argument(
        "--encoder-heads-list",
        type=lambda x: utils.eval_str_list(x, int),
        help="List of integers",
    )
    parser.add_argument(
        "--encoder-mlp-gmlp-heads-list",
        type=lambda x: utils.eval_str_list(x, int),
        help="List of integers",
    )
    parser.add_argument(
        "--decoder-heads-list",
        type=lambda x: utils.eval_str_list(x, int),
        help="List of integers",
    )
    parser.add_argument(
        "--decoder-mlp-gmlp-heads-list",
        type=lambda x: utils.eval_str_list(x, int),
        help="List of integers",
    )
    parser.add_argument(
        "--encoder-conv-dim-list",
        type=lambda x: utils.eval_str_list(x, int),
        help="List of integers",
    )
    parser.add_argument(
        "--decoder-conv-dim-list",
        type=lambda x: utils.eval_str_list(x, int),
        help="List of integers",
    )
    parser.add_argument(
        "--encoder-conv-use-glu-list",
        type=lambda x: utils.eval_str_list(x, int),
        help='List of ints, e.g. "[0, 1]"',
    )
    parser.add_argument(
        "--decoder-conv-use-glu-list",
        type=lambda x: utils.eval_str_list(x, int),
        help="List of integers",
    )
    parser.add_argument(
        "--encoder-decoder-x-heads-list",
        type=lambda x: utils.eval_str_list(x, int),
        help="List of integers",
    )
    parser.add_argument("--weight-softmax", default=True, type=utils.eval_bool)
    parser.add_argument(
        "--weight-dropout",
        type=float,
        metavar="D",
        help="dropout probability for conv weights",
    )
    parser.add_argument(
        "--max-repeats",
        type=int,
        metavar="N",
        help="Max possible UT repeats",
    )

  @classmethod
  def build_model(cls, args, task):
    """Build a new model instance."""

    # make sure all arguments are present in older models
    base_architecture(args)

    if not safe_hasattr(args, "max_source_positions"):
      args.max_source_positions = 1024
    if not safe_hasattr(args, "max_target_positions"):
      args.max_target_positions = 1024

    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

    def build_embedding(dictionary, embed_dim, path=None):
      num_embeddings = len(dictionary)
      padding_idx = dictionary.pad()
      emb = Embedding(num_embeddings, embed_dim, padding_idx)
      # if provided, load from preloaded dictionaries
      if path:
        embed_dict = utils.parse_embedding(path)
        utils.load_embedding(embed_dict, dictionary, emb)
      return emb

    if args.share_all_embeddings:
      if src_dict != tgt_dict:
        raise RuntimeError(
            "--share-all-embeddings requires a joined dictionary"
        )
      if args.encoder_embed_dim != args.decoder_embed_dim:
        raise RuntimeError(
            "--share-all-embeddings requires --encoder-embed-dim to match"
            " --decoder-embed-dim"
        )
      if args.decoder_embed_path and (
          args.decoder_embed_path != args.encoder_embed_path
      ):
        raise RuntimeError(
            "--share-all-embeddings not compatible with --decoder-embed-path"
        )
      encoder_embed_tokens = build_embedding(
          src_dict, args.encoder_embed_dim, args.encoder_embed_path
      )
      decoder_embed_tokens = encoder_embed_tokens
      args.share_decoder_input_output_embed = True
    else:
      encoder_embed_tokens = build_embedding(
          src_dict, args.encoder_embed_dim, args.encoder_embed_path
      )
      decoder_embed_tokens = build_embedding(
          tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
      )

    encoder = LasagnaEncoder(args, src_dict, encoder_embed_tokens)
    decoder = LasagnaEncoder(args, tgt_dict, decoder_embed_tokens)
    return LasagnaModel(encoder, decoder)


class LasagnaEncoder(FairseqEncoder):  # pylint: disable=g-classes-have-attributes
  """Lasagna encoder consisting of *args.encoder_layers* layers.

  Args:
      args (argparse.Namespace): parsed command-line arguments
      dictionary (~fairseq.data.Dictionary): encoding dictionary
      embed_tokens (torch.nn.Embedding): input embedding
  """

  def __init__(self, args, dictionary, embed_tokens):
    super().__init__(dictionary)
    self.dropout_module = FairseqDropout(
        args.dropout, module_name=self.__class__.__name__
    )

    embed_dim = embed_tokens.embedding_dim
    self.padding_idx = embed_tokens.padding_idx
    self.max_source_positions = args.max_source_positions

    self.embed_tokens = embed_tokens
    self.embed_scale = math.sqrt(embed_dim)
    self.embed_positions = (
        PositionalEmbedding(  # pylint: disable=g-long-ternary
            args.max_source_positions,
            embed_dim,
            self.padding_idx,
            learned=args.encoder_learned_pos,
        )
        if not args.no_token_positional_embeddings
        else None
    )

    self.layers = nn.ModuleList([])
    for i in range(args.encoder_layers):
      self.layers.append(
          LasagnaEncoderLayer(
              args,
              kernel_size=args.encoder_kernel_size_list[i],
              layer_type=args.encoder_layer_type_list[i],
              conv_type=args.encoder_conv_type_list[i],
              heads=args.encoder_heads_list[i],
              conv_dim=args.encoder_conv_dim_list[i],
              conv_use_glu=bool(args.encoder_conv_use_glu_list[i]),
              mlp_type=args.encoder_mlp_type_list[i],
              mlp_gmlp_heads=args.encoder_mlp_gmlp_heads_list[i],
              padding_idx=self.padding_idx,
          )
      )
    self.register_buffer("version", torch.Tensor([2]))
    self.normalize = args.encoder_normalize_before
    if self.normalize:
      self.layer_norm = LayerNorm(embed_dim)

  def forward(self, src_tokens, **unused):
    """Forward propagation.

    Args:
      src_tokens (LongTensor): tokens in the source language of shape
            `(batch, src_len)`
      **unused: arguments which are not used, for API compatibility.

    Returns:
        dict:
            - **encoder_out** (Tensor): the last encoder layer's output of
              shape `(src_len, batch, embed_dim)`
            - **encoder_padding_mask** (ByteTensor): the positions of
              padding elements of shape `(batch, src_len)`
    """
    # embed tokens and positions
    x = self.embed_scale * self.embed_tokens(src_tokens)
    if self.embed_positions is not None:
      x += self.embed_positions(src_tokens)
    x = self.dropout_module(x)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    # compute padding mask
    encoder_padding_mask = src_tokens.eq(self.padding_idx)
    if not encoder_padding_mask.any():
      encoder_padding_mask = None

    # encoder layers
    for layer in self.layers:
      x = layer(
          x,
          encoder_padding_mask,
      )

    if self.normalize:
      x = self.layer_norm(x)

    return {
        "encoder_out": x,  # T x B x C
        "encoder_padding_mask": encoder_padding_mask,  # B x T
    }

  def reorder_encoder_out(self, encoder_out, new_order):
    """Reorder encoder output according to *new_order*.

    Args:
        encoder_out: output from the ``forward()`` method
        new_order (LongTensor): desired order

    Returns:
        *encoder_out* rearranged according to *new_order*
    """
    if encoder_out["encoder_out"] is not None:
      encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
          1, new_order
      )
    if encoder_out["encoder_padding_mask"] is not None:
      encoder_out["encoder_padding_mask"] = encoder_out[
          "encoder_padding_mask"
      ].index_select(0, new_order)
    return encoder_out

  def max_positions(self):
    """Maximum input length supported by the encoder."""
    if self.embed_positions is None:
      return self.max_source_positions
    return min(self.max_source_positions, self.embed_positions.max_positions)


class LasagnaDecoder(FairseqIncrementalDecoder):  # pylint: disable=g-classes-have-attributes
  """Lasagna decoder consisting of *args.decoder_layers* layers.

  Args:
      args (argparse.Namespace): parsed command-line arguments
      dictionary (~fairseq.data.Dictionary): decoding dictionary
      embed_tokens (torch.nn.Embedding): output embedding
      no_encoder_attn (bool, optional): whether to attend to encoder outputs.
          Default: ``False``
  """

  def __init__(
      self,
      args,
      dictionary,
      embed_tokens,
      no_encoder_attn=False,
      final_norm=True,
  ):
    super().__init__(dictionary)
    self.dropout_module = FairseqDropout(
        args.dropout, module_name=self.__class__.__name__
    )
    self.share_input_output_embed = args.share_decoder_input_output_embed

    input_embed_dim = embed_tokens.embedding_dim
    embed_dim = args.decoder_embed_dim
    output_embed_dim = args.decoder_output_dim

    padding_idx = embed_tokens.padding_idx
    self.max_target_positions = args.max_target_positions

    self.embed_tokens = embed_tokens
    self.embed_scale = math.sqrt(embed_dim)

    self.project_in_dim = (
        Linear(input_embed_dim, embed_dim, bias=False)
        if embed_dim != input_embed_dim
        else None
    )

    self.embed_positions = (
        PositionalEmbedding(  # pylint: disable=g-long-ternary
            args.max_target_positions,
            embed_dim,
            padding_idx,
            learned=args.decoder_learned_pos,
        )
        if not args.no_token_positional_embeddings
        else None
    )

    self.layers = nn.ModuleList([])
    for i in range(args.decoder_layers):
      self.layers.append(
          LasagnaDecoderLayer(
              args,
              kernel_size=args.decoder_kernel_size_list[i],
              layer_type=args.decoder_layer_type_list[i],
              conv_type=args.decoder_conv_type_list[i],
              heads=args.decoder_heads_list[i],
              conv_dim=args.decoder_conv_dim_list[i],
              conv_use_glu=bool(args.decoder_conv_use_glu_list[i]),
              mlp_type=args.decoder_mlp_type_list[i],
              mlp_gmlp_heads=args.decoder_mlp_gmlp_heads_list[i],
              x_heads=args.encoder_decoder_x_heads_list[i],
          )
      )

    self.adaptive_softmax = None

    self.project_out_dim = (
        Linear(embed_dim, output_embed_dim, bias=False)
        if embed_dim != output_embed_dim and not args.tie_adaptive_weights
        else None
    )

    if args.adaptive_softmax_cutoff is not None:
      self.adaptive_softmax = AdaptiveSoftmax(
          len(dictionary),
          output_embed_dim,
          utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
          dropout=args.adaptive_softmax_dropout,
          adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
          factor=args.adaptive_softmax_factor,
          tie_proj=args.tie_adaptive_proj,
      )
    elif not self.share_input_output_embed:
      self.embed_out = nn.Parameter(
          torch.Tensor(len(dictionary), output_embed_dim)
      )
      nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim**-0.5)
    self.register_buffer("version", torch.Tensor([2]))
    self.normalize = args.decoder_normalize_before and final_norm
    if self.normalize:
      self.layer_norm = LayerNorm(embed_dim)

  def forward(
      self,
      prev_output_tokens,
      encoder_out=None,
      incremental_state=None,
      **kwargs,
  ):
    """Does a forward step.

    Args:

        prev_output_tokens (LongTensor): previous decoder outputs of shape
            `(batch, tgt_len)`, for teacher forcing
        encoder_out (Tensor, optional): output from the encoder, used for
            encoder-side attention
        incremental_state (dict): dictionary used for storing state during
            :ref:`Incremental decoding`
        **kwargs: ignored.

    Returns:
        tuple:
            - the last decoder layer's output of shape `(batch, tgt_len,
              vocab)`
            - the last decoder layer's attention weights of shape `(batch,
              tgt_len, src_len)`
    """
    # embed positions
    positions = (
        self.embed_positions(  # pylint: disable=g-long-ternary
            prev_output_tokens,
            incremental_state=incremental_state,
        )
        if self.embed_positions is not None
        else None
    )

    if incremental_state is not None:
      prev_output_tokens = prev_output_tokens[:, -1:]
      if positions is not None:
        positions = positions[:, -1:]

    # embed tokens and positions
    x = self.embed_scale * self.embed_tokens(prev_output_tokens)

    if self.project_in_dim is not None:
      x = self.project_in_dim(x)

    if positions is not None:
      x += positions
    x = self.dropout_module(x)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)
    attn = None

    inner_states = [x]

    # decoder layers
    for layer in self.layers:
      x, attn = layer(
          x,
          encoder_out["encoder_out"] if encoder_out is not None else None,
          encoder_out["encoder_padding_mask"]
          if encoder_out is not None
          else None,
          incremental_state,
      )
      inner_states.append(x)

    if self.normalize:
      x = self.layer_norm(x)

    # T x B x C -> B x T x C
    x = x.transpose(0, 1)

    if self.project_out_dim is not None:
      x = self.project_out_dim(x)

    if self.adaptive_softmax is None:
      # project back to size of vocabulary
      if self.share_input_output_embed:
        x = F.linear(x, self.embed_tokens.weight)
      else:
        x = F.linear(x, self.embed_out)

    return x, {"attn": attn, "inner_states": inner_states}

  def max_positions(self):
    """Maximum output length supported by the decoder."""
    if self.embed_positions is None:
      return self.max_target_positions
    return min(self.max_target_positions, self.embed_positions.max_positions)

  def buffered_future_mask(self, tensor):
    dim = tensor.size(0)
    if (
        not hasattr(self, "_future_mask")
        or self._future_mask is None
        or self._future_mask.device != tensor.device
    ):
      self._future_mask = torch.triu(
          utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
      )
    if self._future_mask.size(0) < dim:
      self._future_mask = torch.triu(
          utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
      )
    return self._future_mask[:dim, :dim]


class LasagnaEncoderLayer(nn.Module):  # pylint: disable=g-classes-have-attributes
  """Encoder layer block.

  Args:
      args (argparse.Namespace): parsed command-line arguments
      kernel_size: kernel size of the convolution
  """

  def build_conv(self, args):
    if self.conv_use_glu:
      self.linear1 = Linear(self.embed_dim, 2 * self.conv_dim)
      self.act = nn.GLU()
    else:
      self.linear1 = Linear(self.embed_dim, self.conv_dim)
      self.act = None

    kernel_size = self.kernel_size

    padding_l = (
        kernel_size // 2
        if kernel_size % 2 == 1
        else ((kernel_size - 1) // 2, kernel_size // 2)
    )

    if self.conv_type == "lightweight":
      self.conv = LightweightConv(
          self.conv_dim,
          kernel_size,
          padding_l=padding_l,
          weight_softmax=args.weight_softmax,
          num_heads=self.heads,
          weight_dropout=args.weight_dropout,
      )
    elif self.conv_type == "dynamic":
      self.conv = DynamicConv(
          self.conv_dim,
          kernel_size,
          padding_l=padding_l,
          weight_softmax=args.weight_softmax,
          num_heads=self.heads,
          weight_dropout=args.weight_dropout,
      )
    else:
      raise NotImplementedError
    self.linear2 = Linear(self.conv_dim, self.embed_dim)

  def build_gmlp(self, args):
    kernel_size = self.kernel_size
    padding_l = (
        kernel_size // 2
        if kernel_size % 2 == 1
        else ((kernel_size - 1) // 2, kernel_size // 2)
    )
    self.linear1 = Linear(self.embed_dim, 2 * self.conv_dim)
    self.act = nn.GELU()

    if self.conv_type == "lightweight":
      self.conv = LightweightConv(
          self.conv_dim,
          kernel_size,
          padding_l=padding_l,
          weight_softmax=args.weight_softmax,
          num_heads=self.heads,
          weight_dropout=args.weight_dropout,
      )
    elif self.conv_type == "dynamic":
      self.conv = DynamicConv(
          self.conv_dim,
          kernel_size,
          padding_l=padding_l,
          weight_softmax=args.weight_softmax,
          num_heads=self.heads,
          weight_dropout=args.weight_dropout,
      )
    else:
      raise NotImplementedError
    self.linear2 = Linear(self.conv_dim, self.embed_dim)

  def build_self_attn(self, args):
    self.self_attn = MultiheadAttention(
        self.embed_dim,
        self.heads,
        dropout=args.attention_dropout,
        self_attention=True,
    )

  def build_mlp(self, args):
    self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
    self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
    self.num_layer_norms = 2

  def build_gmlp_for_mlp(self, args):
    kernel_size = self.kernel_size
    padding_l = (
        kernel_size // 2
        if kernel_size % 2 == 1
        else ((kernel_size - 1) // 2, kernel_size // 2)
    )
    self.mlp_lin1 = Linear(self.embed_dim, 2 * args.encoder_ffn_embed_dim)
    self.mlp_act = nn.GELU()

    self.mlp_conv_dim = args.encoder_ffn_embed_dim

    if self.mlp_type == "gmlp_l":
      self.mlp_conv = LightweightConv(
          self.mlp_conv_dim,
          kernel_size,
          padding_l=padding_l,
          weight_softmax=args.weight_softmax,
          num_heads=self.mlp_gmlp_heads,
          weight_dropout=args.weight_dropout,
      )
    elif self.mlp_type == "gmlp_d":
      self.mlp_conv = DynamicConv(
          self.mlp_conv_dim,
          kernel_size,
          padding_l=padding_l,
          weight_softmax=args.weight_softmax,
          num_heads=self.mlp_gmlp_heads,
          weight_dropout=args.weight_dropout,
      )
    else:
      raise NotImplementedError
    self.mlp_lin2 = Linear(self.mlp_conv_dim, self.embed_dim)
    self.num_layer_norms = 2

  def __init__(
      self,
      args,
      kernel_size=0,
      layer_type = None,
      conv_type = None,
      heads = 0,
      conv_dim = 0,
      num_experts = 0,
      conv_use_glu = False,
      mlp_type = None,
      mlp_gmlp_heads = 0,
      padding_idx = 0,
  ):
    super().__init__()

    assert layer_type in ("conv", "gmlp", "self")
    assert mlp_type in ("no", "mlp", "gmlp_l", "gmlp_d")
    assert conv_type in ("lightweight", "dynamic")
    assert heads > 0
    assert conv_dim > 0

    self.kernel_size = kernel_size
    self.embed_dim = args.encoder_embed_dim
    self.conv_dim = conv_dim
    self.conv_use_glu = conv_use_glu
    self.layer_type = layer_type
    self.conv_type = conv_type
    self.heads = heads
    self.num_experts = num_experts
    self.mlp_type = mlp_type
    self.mlp_gmlp_heads = mlp_gmlp_heads
    self.padding_idx = padding_idx

    if layer_type == "conv":
      self.build_conv(args)
    elif layer_type == "gmlp":
      self.build_gmlp(args)
    else:
      assert layer_type == "self"
      self.build_self_attn(args)

    self.dropout_module = FairseqDropout(
        args.dropout, module_name=self.__class__.__name__
    )
    self.relu_dropout_module = FairseqDropout(
        args.relu_dropout, module_name=self.__class__.__name__
    )
    self.input_dropout_module = FairseqDropout(
        args.input_dropout, module_name=self.__class__.__name__
    )
    self.normalize_before = args.encoder_normalize_before

    self.num_layer_norms = 1

    # mlp-building needs to be called here and needs to update the layer_norms
    # by 1 if the layer is instantiated
    if self.mlp_type == "mlp":
      self.build_mlp(args)
    elif self.mlp_type in ("gmlp_d", "gmlp_l"):
      self.build_gmlp_for_mlp(args)
    self.layer_norms = nn.ModuleList(
        [LayerNorm(self.embed_dim) for _ in range(self.num_layer_norms)]
    )

  def apply_gmlp(self, x, encoder_padding_mask):
    residual = x
    x = self.maybe_layer_norm(0, x, before=True)
    x = self.input_dropout_module(x)
    x = self.linear1(x)

    x = self.act(x)
    if encoder_padding_mask is not None:
      x = x.masked_fill(encoder_padding_mask.transpose(0, 1).unsqueeze(2), 0)
    u, v = torch.split(x, [self.conv_dim, self.conv_dim], dim=2)
    v = self.conv(v)
    x = u * (1.0 + v)
    x = self.linear2(x)
    x = self.dropout_module(x)
    x = residual + x
    x = self.maybe_layer_norm(0, x, after=True)
    return x

  def apply_conv(self, x, encoder_padding_mask):
    residual = x
    x = self.maybe_layer_norm(0, x, before=True)
    x = self.input_dropout_module(x)
    x = self.linear1(x)
    if self.act is not None:
      x = self.act(x)
    if encoder_padding_mask is not None:
      x = x.masked_fill(encoder_padding_mask.transpose(0, 1).unsqueeze(2), 0)
    x = self.conv(x)
    x = self.linear2(x)
    x = self.dropout_module(x)
    x = residual + x
    x = self.maybe_layer_norm(0, x, after=True)
    return x

  def apply_self_attn(self, x, encoder_padding_mask):
    residual = x
    x = self.maybe_layer_norm(0, x, before=True)
    x, _ = self.self_attn(
        query=x,
        key=x,
        value=x,
        key_padding_mask=encoder_padding_mask,
        need_weights=False,
    )
    x = self.dropout_module(x)
    x = residual + x
    x = self.maybe_layer_norm(0, x, after=True)
    return x

  def apply_mlp(self, x):
    residual = x
    x = self.maybe_layer_norm(1, x, before=True)
    x = F.relu(self.fc1(x))
    x = self.relu_dropout_module(x)
    x = self.fc2(x)
    x = self.dropout_module(x)
    x = residual + x
    x = self.maybe_layer_norm(1, x, after=True)
    return x

  def apply_gmlp_mlp(self, x, encoder_padding_mask):
    residual = x
    x = self.maybe_layer_norm(1, x, before=True)
    x = self.relu_dropout_module(x)
    x = self.mlp_lin1(x)

    x = self.mlp_act(x)
    if encoder_padding_mask is not None:
      x = x.masked_fill(encoder_padding_mask.transpose(0, 1).unsqueeze(2), 0)
    u, v = torch.split(x, [self.mlp_conv_dim, self.mlp_conv_dim], dim=2)
    v = self.mlp_conv(v)
    x = u * (1.0 + v)
    x = self.mlp_lin2(x)
    x = self.dropout_module(x)
    x = residual + x
    x = self.maybe_layer_norm(1, x, after=True)
    return x

  def forward(self, x, encoder_padding_mask=None):
    """Forward propagation.

    Args:

        x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
        encoder_padding_mask (ByteTensor): binary ByteTensor of shape
            `(batch, src_len)` where padding elements are indicated by ``1``.

    Returns:
        encoded output of shape `(batch, src_len, embed_dim)`
    """

    if self.layer_type == "conv":
      x = self.apply_conv(x, encoder_padding_mask)
    elif self.layer_type == "gmlp":
      x = self.apply_gmlp(x, encoder_padding_mask)
    else:
      assert self.layer_type == "self"
      x = self.apply_self_attn(x, encoder_padding_mask)

    if self.mlp_type == "no":
      x = x  # pylint: disable=self-assigning-variable
    elif self.mlp_type == "mlp":
      x = self.apply_mlp(x)
    elif self.mlp_type in ("gmlp_l", "gmlp_d",):
      x = self.apply_gmlp_mlp(x, encoder_padding_mask)
    return x

  def maybe_layer_norm(self, i, x, before=False, after=False):
    assert before ^ after
    if after ^ self.normalize_before:
      return self.layer_norms[i](x)
    else:
      return x

  def extra_repr(self):
    return (
        "kernel_size={}, layer_type={}, conv_type={}, heads={}, conv_dim={},"
        " conv_use_glu={}, mlp_type={}, num_layer_norms={}, dropout={},"
        " relu_dropout={}, input_dropout={}, normalize_before={},"
        " mlp_gmlp_heads={} ".format(
            self.kernel_size,
            self.layer_type,
            self.conv_type,
            self.heads,
            self.conv_dim,
            self.conv_use_glu,
            self.mlp_type,
            self.num_layer_norms,
            self.dropout_module.p,
            self.relu_dropout_module.p,
            self.input_dropout_module.p,
            self.normalize_before,
            self.mlp_gmlp_heads,
        )
    )


class LasagnaDecoderLayer(nn.Module):  # pylint: disable=g-classes-have-attributes
  """Decoder layer block.

  Args:
      args (argparse.Namespace): parsed command-line arguments
      no_encoder_attn (bool, optional): whether to attend to encoder outputs.
          Default: ``False``
      kernel_size: kernel size of the convolution
  """

  def build_conv(self, args):
    if self.conv_use_glu:
      self.linear1 = Linear(self.embed_dim, 2 * self.conv_dim)
      self.act = nn.GLU()
    else:
      self.linear1 = Linear(self.embed_dim, self.conv_dim)
      self.act = None

    kernel_size = self.kernel_size

    padding_l = self.kernel_size - 1

    if self.conv_type == "lightweight":
      self.conv = LightweightConv(
          self.conv_dim,
          kernel_size,
          padding_l=padding_l,
          weight_softmax=args.weight_softmax,
          num_heads=self.heads,
          weight_dropout=args.weight_dropout,
      )
    elif self.conv_type == "dynamic":
      self.conv = DynamicConv(
          self.conv_dim,
          kernel_size,
          padding_l=padding_l,
          weight_softmax=args.weight_softmax,
          num_heads=self.heads,
          weight_dropout=args.weight_dropout,
      )
    else:
      raise NotImplementedError
    self.linear2 = Linear(self.conv_dim, self.embed_dim)

  def build_gmlp(self, args):
    kernel_size = self.kernel_size
    self.linear1 = Linear(self.embed_dim, 2 * self.conv_dim)
    self.act = nn.GELU()

    if self.conv_type == "lightweight":
      self.conv = LightweightConv(
          self.conv_dim,
          kernel_size,
          padding_l=kernel_size - 1,
          weight_softmax=args.weight_softmax,
          num_heads=self.heads,
          weight_dropout=args.weight_dropout,
      )
    elif self.conv_type == "dynamic":
      self.conv = DynamicConv(
          self.conv_dim,
          kernel_size,
          padding_l=kernel_size - 1,
          weight_softmax=args.weight_softmax,
          num_heads=self.heads,
          weight_dropout=args.weight_dropout,
      )
    else:
      raise NotImplementedError
    self.linear2 = Linear(self.conv_dim, self.embed_dim)

  def build_self_attn(self, args):
    self.self_attn = MultiheadAttention(
        self.embed_dim,
        self.heads,
        dropout=args.attention_dropout,
        self_attention=True,
    )

  def build_encoder_attn(self, args):

    if self.x_attention_type == "no":
      self.encoder_attn = None
      self.encoder_attn_layer_norm = None
    else:
      assert self.x_attention_type == "attention"
      self.encoder_attn = MultiheadAttention(
          self.embed_dim,
          self.x_heads,
          dropout=args.attention_dropout,
          encoder_decoder_attention=True,
      )
      self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

  def build_mlp(self, args):
    self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
    self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
    self.final_layer_norm = LayerNorm(self.embed_dim)

  def build_gmlp_mlp(self, args):
    kernel_size = self.kernel_size
    self.mlp_lin1 = Linear(self.embed_dim, 2 * args.decoder_ffn_embed_dim)
    self.mlp_act = nn.GELU()

    self.mlp_conv_dim = args.decoder_ffn_embed_dim

    if self.mlp_type == "gmlp_l":
      self.mlp_conv = LightweightConv(
          self.mlp_conv_dim,
          kernel_size,
          padding_l=kernel_size - 1,
          weight_softmax=args.weight_softmax,
          num_heads=self.mlp_gmlp_heads,
          weight_dropout=args.weight_dropout,
      )
    elif self.mlp_type == "gmlp_d":
      self.mlp_conv = DynamicConv(
          self.mlp_conv_dim,
          kernel_size,
          padding_l=kernel_size - 1,
          weight_softmax=args.weight_softmax,
          num_heads=self.mlp_gmlp_heads,
          weight_dropout=args.weight_dropout,
      )
    else:
      raise NotImplementedError
    self.mlp_lin2 = Linear(self.mlp_conv_dim, self.embed_dim)
    self.final_layer_norm = LayerNorm(self.embed_dim)

  def __init__(
      self,
      args,
      kernel_size=0,
      layer_type="",
      conv_type="",
      heads = 0,
      conv_dim = 0,
      num_experts = 0,
      num_encoder_decoder_experts = 0,
      conv_use_glu = False,
      mlp_type = None,
      x_attention_type = "",
      x_heads = 0,
      mlp_gmlp_heads = 0,
  ):
    super().__init__()
    assert mlp_type in ("no", "mlp", "gmlp_l", "gmlp_d",)
    assert layer_type in ("conv", "gmlp", "self")
    assert conv_type in ("lightweight", "dynamic",)
    assert x_attention_type in ("no", "attention",)
    assert heads > 0
    assert conv_dim > 0

    self.kernel_size = kernel_size
    self.embed_dim = args.encoder_embed_dim
    self.conv_dim = conv_dim
    self.conv_use_glu = conv_use_glu
    self.layer_type = layer_type
    self.conv_type = conv_type
    self.heads = heads
    self.x_attention_type = x_attention_type
    self.mlp_type = mlp_type
    self.x_heads = x_heads
    self._future_mask = torch.empty(0)
    self.embed_dim = args.decoder_embed_dim
    self.mlp_gmlp_heads = mlp_gmlp_heads

    if layer_type == "conv":
      self.build_conv(args)
    elif layer_type == "gmlp":
      self.build_gmlp(args)
    else:
      assert layer_type == "self"
      self.build_self_attn(args)

    self.build_encoder_attn(args)

    self.dropout_module = FairseqDropout(
        args.dropout, module_name=self.__class__.__name__
    )
    self.relu_dropout_module = FairseqDropout(
        args.relu_dropout, module_name=self.__class__.__name__
    )
    self.input_dropout_module = FairseqDropout(
        args.input_dropout, module_name=self.__class__.__name__
    )
    self.normalize_before = args.decoder_normalize_before

    self.conv_layer_norm = LayerNorm(self.embed_dim)

    if self.mlp_type == "mlp":
      self.build_mlp(args)
    elif self.mlp_type in ("gmlp_d", "gmlp_l"):
      self.build_gmlp_mlp(args)

    self.need_attn = True

  def buffered_future_mask(self, tensor):
    dim = tensor.size(0)
    # self._future_mask.device != tensor.device is not working in TorchScript.
    # This is a workaround.
    if (
        self._future_mask.size(0) == 0
        or (not self._future_mask.device == tensor.device)  # pylint: disable=g-comparison-negation
        or self._future_mask.size(0) < dim
    ):
      self._future_mask = torch.triu(
          utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
      )
    self._future_mask = self._future_mask.to(tensor)
    return self._future_mask[:dim, :dim]

  def apply_conv(self, x, incremental_state, prev_conv_state):
    residual = x
    x = self.maybe_layer_norm(self.conv_layer_norm, x, before=True)
    if prev_conv_state is not None:
      if incremental_state is None:
        incremental_state = {}
      self.conv._set_input_buffer(incremental_state, prev_conv_state)  # pylint: disable=protected-access
    x = self.input_dropout_module(x)
    x = self.linear1(x)
    if self.act is not None:
      x = self.act(x)
    x = self.conv(x, incremental_state=incremental_state)
    x = self.linear2(x)
    x = self.dropout_module(x)
    x = residual + x
    x = self.maybe_layer_norm(self.conv_layer_norm, x, after=True)
    return x

  def apply_gmlp(self, x, incremental_state, prev_conv_state):
    residual = x
    x = self.maybe_layer_norm(self.conv_layer_norm, x, before=True)
    if prev_conv_state is not None:
      if incremental_state is None:
        incremental_state = {}
      self.conv._set_input_buffer(incremental_state, prev_conv_state)  # pylint: disable=protected-access
    x = self.input_dropout_module(x)
    x = self.linear1(x)
    x = self.act(x)
    u, v = torch.split(x, [self.conv_dim, self.conv_dim], dim=2)
    v = self.conv(v, incremental_state=incremental_state)
    x = u * (1.0 + v)
    x = self.linear2(x)
    x = self.dropout_module(x)
    x = residual + x
    x = self.maybe_layer_norm(self.conv_layer_norm, x, after=True)
    return x

  def apply_self_attn(
      self,
      x,
      incremental_state,
      override_self_attn_mask,
      override_key_padding_mask,
  ):
    residual = x
    x = self.maybe_layer_norm(self.conv_layer_norm, x, before=True)
    if override_self_attn_mask is not None:
      self_attn_mask = override_self_attn_mask
    elif incremental_state is None:
      self_attn_mask = self.buffered_future_mask(x)
    else:
      self_attn_mask = None

    x, _ = self.self_attn(
        query=x,
        key=x,
        value=x,
        incremental_state=incremental_state,
        attn_mask=self_attn_mask,
        key_padding_mask=override_key_padding_mask,
        need_weights=False,
    )

    x = self.dropout_module(x)
    x = residual + x
    x = self.maybe_layer_norm(self.conv_layer_norm, x, after=True)
    return x

  def apply_x_attention(
      self,
      x,
      encoder_out,
      encoder_padding_mask,
      incremental_state,
      prev_attn_state,
  ):
    residual = x
    x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
    if self.x_attention_type == "attention":
      if prev_attn_state is not None:
        if incremental_state is None:
          incremental_state = {}
        prev_key, prev_value = prev_attn_state
        saved_state = {"prev_key": prev_key, "prev_value": prev_value}
        self.encoder_attn._set_input_buffer(incremental_state, saved_state)  # pylint: disable=protected-access
      # print('Debug for Leven: encoder_out', encoder_out, encoder_out.size())
      x, attn = self.encoder_attn(
          query=x,
          key=encoder_out,
          value=encoder_out,
          key_padding_mask=encoder_padding_mask,
          incremental_state=incremental_state,
          static_kv=True,
          need_weights=(not self.training and self.need_attn),
      )
    else:
      return residual, None  # A no op
    x = self.dropout_module(x)
    x = residual + x
    x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

    return x, attn

  def apply_mlp(self, x):
    residual = x
    x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
    x = F.relu(self.fc1(x))
    x = self.relu_dropout_module(x)
    x = self.fc2(x)
    x = self.dropout_module(x)
    x = residual + x
    x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
    return x

  def apply_mlp_gmlp(self, x, incremental_state, prev_conv_state):
    residual = x
    x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
    if prev_conv_state is not None:
      if incremental_state is None:
        incremental_state = {}
      self.mlp_conv._set_input_buffer(incremental_state, prev_conv_state)  # pylint: disable=protected-access
    x = self.relu_dropout_module(x)
    x = self.mlp_lin1(x)
    x = self.mlp_act(x)
    u, v = torch.split(x, [self.mlp_conv_dim, self.mlp_conv_dim], dim=2)
    v = self.mlp_conv(v, incremental_state=incremental_state)
    x = u * (1.0 + v)
    x = self.mlp_lin2(x)
    x = self.dropout_module(x)
    x = residual + x
    x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
    return x

  def forward(
      self,
      x,
      encoder_out,
      encoder_padding_mask,
      incremental_state,
      prev_conv_state=None,
      prev_attn_state=None,
      conv_mask=None,
      conv_padding_mask=None,
      override_self_attn_mask=None,
      override_key_padding_mask=None,
  ):
    """Applies forward step.

    Args:

        x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
        encoder_out (Tensor): output of the Encoder.
        encoder_padding_mask (ByteTensor): binary ByteTensor of shape
            `(batch, src_len)` where padding elements are indicated by ``1``.
        incremental_state: Used during auto-regressive decoding as a cache.
        prev_conv_state: Can be used to override the convolutional
          result from previous time step.
        prev_attn_state: Not used, can be used to override the attention
          result from previous time step.
        conv_mask: Optional mask to use with convolutions.
        conv_padding_mask: Optional masking to use with convolutions.
        override_self_attn_mask: overrides the self-attention mask.
        override_key_padding_mask: overrides the padding mask for the keys.

    Returns:
        encoded output of shape `(batch, src_len, embed_dim)`
    """

    if self.layer_type == "gmlp":
      x = self.apply_gmlp(x, incremental_state, prev_conv_state)
    elif self.layer_type == "conv":
      x = self.apply_conv(x, incremental_state, prev_conv_state)
    else:
      x = self.apply_self_attn(
          x,
          incremental_state,
          override_self_attn_mask=override_self_attn_mask,
          override_key_padding_mask=override_key_padding_mask,
      )

    x, attn = self.apply_x_attention(
        x, encoder_out, encoder_padding_mask, incremental_state, prev_attn_state
    )

    if self.mlp_type == "mlp":
      x = self.apply_mlp(x)
    elif self.mlp_type in ("gmlp_d", "gmlp_l"):
      x = self.apply_mlp_gmlp(x, incremental_state, prev_conv_state)

    return x, attn

  def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
    assert before ^ after
    if after ^ self.normalize_before:
      return layer_norm(x)
    else:
      return x

  def make_generation_fast_(self, need_attn=False, **kwargs):
    self.need_attn = need_attn

  def extra_repr(self):
    return """kernel_size={}, layer_type={}, conv_type={}, heads={}, conv_dim={}, mlp_type={}, conv_use_glu={},
        x_attention_type={}, x_heads= {}, mlp_gmlp_heads = {},
        dropout={}, relu_dropout={}, input_dropout={}, normalize_before={}""".format(
        self.kernel_size,
        self.layer_type,
        self.conv_type,
        self.heads,
        self.conv_dim,
        self.conv_use_glu,
        self.mlp_type,
        self.x_attention_type,
        self.x_heads,
        self.mlp_gmlp_heads,
        self.dropout_module.p,
        self.relu_dropout_module.p,
        self.input_dropout_module.p,
        self.normalize_before,
    )


def Embedding(num_embeddings, embedding_dim, padding_idx):  # pylint: disable=invalid-name
  m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
  nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
  nn.init.constant_(m.weight[padding_idx], 0)
  return m


def Linear(in_features, out_features, bias=True):  # pylint: disable=invalid-name
  m = nn.Linear(in_features, out_features, bias)
  nn.init.xavier_uniform_(m.weight)
  if bias:
    nn.init.constant_(m.bias, 0.0)
  return m


@register_model_architecture("lasagna", "lasagna")
def base_architecture(args):
  """Parameters for base architecture."""
  args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
  args.max_repeats = getattr(args, "max_repeats", 256)
  args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
  args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
  args.encoder_layers = getattr(args, "encoder_layers", 7)
  args.encoder_heads_list = getattr(
      args, "encoder_heads_list", [8 for _ in range(70)]
  )
  args.encoder_mlp_gmlp_heads_list = getattr(
      args, "encoder_mlp_gmlp_heads_list", [16 for _ in range(70)]
  )
  args.encoder_conv_type_list = getattr(
      args, "encoder_conv_type_list", ["lightweight" for _ in range(70)]
  )
  args.encoder_conv_dim_list = getattr(
      args, "encoder_conv_dim_list", [512 for _ in range(70)]
  )
  args.encoder_conv_use_glu_list = getattr(
      args, "encoder_conv_use_glu_list", [0 for _ in range(70)]
  )
  args.encoder_layer_type_list = getattr(
      args, "encoder_layer_type_list", ["gmlp" for _ in range(70)]
  )
  args.encoder_mlp_type_list = getattr(
      args, "encoder_mlp_type_list", ["mlp" for _ in range(70)]
  )
  args.encoder_normalize_before = getattr(
      args, "encoder_normalize_before", False
  )
  args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
  args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
  args.decoder_embed_dim = getattr(
      args, "decoder_embed_dim", args.encoder_embed_dim
  )
  args.decoder_ffn_embed_dim = getattr(
      args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
  )
  args.decoder_layers = getattr(args, "decoder_layers", 6)
  args.decoder_heads_list = getattr(
      args, "decoder_heads_list", [8 for _ in range(60)]
  )
  args.decoder_mlp_gmlp_heads_list = getattr(
      args, "decoder_mlp_gmlp_heads_list", [16 for _ in range(60)]
  )
  args.decoder_conv_type_list = getattr(
      args, "decoder_conv_type_list", ["lightweight" for _ in range(60)]
  )
  args.decoder_conv_dim_list = getattr(
      args, "decoder_conv_dim_list", [512 for _ in range(60)]
  )
  args.decoder_conv_use_glu_list = getattr(
      args, "decoder_conv_use_glu_list", [0 for _ in range(60)]
  )
  args.decoder_layer_type_list = getattr(
      args, "decoder_layer_type_list", ["gmlp" for _ in range(60)]
  )
  args.decoder_mlp_type_list = getattr(
      args, "decoder_mlp_type_list", ["mlp" for _ in range(60)]
  )
  args.decoder_normalize_before = getattr(
      args, "decoder_normalize_before", False
  )
  args.encoder_decoder_x_attention_type_list = getattr(
      args,
      "encoder_decoder_x_attention_type_list",
      ["attention" for _ in range(60)],
  )
  args.encoder_decoder_x_heads_list = getattr(
      args, "encoder_decoder_x_heads_list", [8 for _ in range(60)]
  )
  args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
  args.attention_dropout = getattr(args, "attention_dropout", 0.0)
  args.relu_dropout = getattr(args, "relu_dropout", 0.0)
  args.dropout = getattr(args, "dropout", 0.1)
  args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
  args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
  args.share_decoder_input_output_embed = getattr(
      args, "share_decoder_input_output_embed", False
  )
  args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
  args.no_token_positional_embeddings = getattr(
      args, "no_token_positional_embeddings", False
  )

  args.decoder_output_dim = getattr(
      args, "decoder_output_dim", args.decoder_embed_dim
  )
  args.decoder_input_dim = getattr(
      args, "decoder_input_dim", args.decoder_embed_dim
  )

  args.encoder_conv_dim = getattr(
      args, "encoder_conv_dim", args.encoder_embed_dim
  )
  args.decoder_conv_dim = getattr(
      args, "decoder_conv_dim", args.decoder_embed_dim
  )

  args.encoder_kernel_size_list = getattr(
      args, "encoder_kernel_size_list", [3, 7, 15, 31, 31, 31, 31]
  )
  args.decoder_kernel_size_list = getattr(
      args, "decoder_kernel_size_list", [3, 7, 15, 31, 31, 31]
  )
  if len(args.encoder_kernel_size_list) == 1:
    args.encoder_kernel_size_list = (
        args.encoder_kernel_size_list * args.encoder_layers
    )
  if len(args.decoder_kernel_size_list) == 1:
    args.decoder_kernel_size_list = (
        args.decoder_kernel_size_list * args.decoder_layers
    )
  assert (
      len(args.encoder_kernel_size_list) == args.encoder_layers
  ), "encoder_kernel_size_list doesn't match encoder_layers"
  assert (
      len(args.decoder_kernel_size_list) == args.decoder_layers
  ), "decoder_kernel_size_list doesn't match decoder_layers"
  args.input_dropout = getattr(args, "input_dropout", 0.1)
  args.weight_dropout = getattr(args, "weight_dropout", args.attention_dropout)


@register_model_architecture("lasagna", "lasagna_iwslt_de_en")
def lasagna_iwslt_de_en(args):
  """Parameters for IWSLT dataset."""
  args.encoder_layers = getattr(args, "encoder_layers", 7)
  args.encoder_heads_list = getattr(
      args, "encoder_heads_list", [4 for _ in range(70)]
  )
  args.encoder_mlp_gmlp_heads_list = getattr(
      args, "encoder_mlp_gmlp_heads_list", [8 for _ in range(70)]
  )
  args.decoder_layers = getattr(args, "decoder_layers", 6)
  args.decoder_heads_list = getattr(
      args, "decoder_heads_list", [4 for _ in range(60)]
  )
  args.decoder_mlp_gmlp_heads_list = getattr(
      args, "decoder_mlp_gmlp_heads_list", [8 for _ in range(60)]
  )
  args.encoder_decoder_x_heads_list = getattr(
      args, "encoder_decoder_x_heads_list", [4 for _ in range(60)]
  )
  args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
  args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
  args.attention_dropout = getattr(args, "attention_dropout", 0.1)
  args.weight_dropout = getattr(args, "weight_dropout", 0.1)
  args.input_dropout = getattr(args, "input_dropout", 0.0)
  base_architecture(args)


@register_model_architecture("lasagna", "lasagna_wmt_en_de")
def lasagna_wmt_en_de(args):
  base_architecture(args)


@register_model_architecture("lasagna", "lasagna_wmt_en_de_big")
def lasagna_wmt_en_de_big(args):
  """Parameters for WMT EnDe at model size big."""
  args.attention_dropout = getattr(args, "attention_dropout", 0.1)
  args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
  args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
  args.encoder_heads_list = getattr(
      args, "encoder_heads_list", [16 for _ in range(70)]
  )
  args.encoder_mlp_gmlp_heads_list = getattr(
      args, "encoder_mlp_gmlp_heads_list", [16 for _ in range(70)]
  )
  args.encoder_conv_dim_list = getattr(
      args, "encoder_conv_dim_list", [1024 for _ in range(70)]
  )
  args.encoder_normalize_before = getattr(
      args, "encoder_normalize_before", False
  )
  args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
  args.decoder_heads_list = getattr(
      args, "decoder_heads_list", [16 for _ in range(60)]
  )
  args.decoder_mlp_gmlp_heads_list = getattr(
      args, "decoder_mlp_gmlp_heads_list", [16 for _ in range(70)]
  )
  args.decoder_conv_dim_list = getattr(
      args, "decoder_conv_dim_list", [1024 for _ in range(70)]
  )
  args.encoder_decoder_x_heads_list = getattr(
      args, "encoder_decoder_x_heads_list", [16 for _ in range(70)]
  )
  args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
  args.dropout = getattr(args, "dropout", 0.3)
  base_architecture(args)


@register_model_architecture("lasagna", "lasagna_wmt_zh_en_big")
def lasagna_wmt_zh_en_big(args):
  """Parameters for WMT ZhEn at model size Big."""
  args.dropout = getattr(args, "dropout", 0.2)
  args.attention_dropout = getattr(args, "attention_dropout", 0.2)
  args.weight_dropout = getattr(args, "weight_dropout", 0.2)
  lasagna_wmt_en_de_big(args)
