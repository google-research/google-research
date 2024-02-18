"""A wrapper for CLIP model to support forward with a list of text inputs"""

import clip
from clip.model import CLIP, VisionTransformer
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def forward_clip_single(model, image, text, H, W):
    """Forward a single text input.

    Args:
        model (CLIPWrapper or CLIP): the CLIP model.
        image (torch.Tensor): the image tensor.
        text (List[str]): the text input.
        H (int): the height of the image.
        W (int): the width of the image.

    Returns:
        torch.Tensor: the logits.
    """
    if isinstance(text, str):
        text = [text]
    text_tokens = clip.tokenize(text).to(image.device)
    text_prediction = model(image, text_tokens, H, W)
    return text_prediction.detach().cpu()


def forward_clip(model, image, text, H, W):
    """Forward a list of text inputs.

    Args:
        model (CLIPWrapper or CLIP): the CLIP model.
        image (torch.Tensor): the image tensor.
        text (List[str]): the text input.
        H (int): the height of the image.
        W (int): the width of the image.

    Returns:
        torch.Tensor: the logits.
    """
    if isinstance(text[0], list):
        text_prediction = torch.stack(
            [forward_clip_single(model, image, t, H, W) for t in text], dim=0)
        text_prediction = torch.sum(text_prediction, dim=0)
        text_prediction = F.softmax(text_prediction.float(), dim=-1)
    else:
        text_prediction = forward_clip_single(model, image, text, H, W)
    return text_prediction.float()


def upsample_position_embedding(embed, new_size):
    """Upsample the pretrained embedding to a higher resolution.

    Args:
        embed (torch.Tensor): the pretrained embedding.
        new_size (Tuple[int, int]): the new size of the embedding.

    Returns:
        torch.Tensor: the upsampled embedding.
    """
    # emb size NxD
    first = embed[:1, :]
    embed = embed[1:, :]
    N, D = embed.size(0), embed.size(1)
    size = int(np.sqrt(N))
    assert size * size == N
    # new_size = size * self.upsample
    embed = embed.permute(1, 0)
    embed = embed.view(1, D, size, size).contiguous()
    embed = F.upsample(embed, size=new_size, mode='bilinear',)
    embed = embed.view(D, -1).contiguous()
    embed = embed.permute(1, 0)
    embed = torch.cat([first, embed], 0)
    embed = nn.parameter.Parameter(embed.half())
    return embed


class CustomBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        for k, v in vars(block).items():
            setattr(self, k, v)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype,
            device=x.device) if self.attn_mask is not None else None
        self.attn = self.attn.to(dtype=x.dtype, device=x.device)
        # Setting need_weights to True also returns the attention weights
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor):
        # attn_output: (L,N,E), attn_weight: (N,L,L)
        attn_output, attn_weight = self.attention(self.ln_1(x))
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weight


class CustomTransformer(nn.Module):
    """A customized Transformer to support CAM calculation."""

    def __init__(self, transformer: nn.Module):
        """Initialize the wrapper.

        Args:
            transformer (nn.Module): the Transformer to be wrapped.

        """
        super().__init__()
        for k, v in vars(transformer).items():
            setattr(self, k, v)

        self.resblocks = nn.Sequential(
            *[CustomBlock(block) for block in self.resblocks])

    def forward(self, x: torch.Tensor):
        attn_weights = []
        with torch.no_grad():
            layers = self.layers if x.shape[0] == 77 else self.layers-1
            for i in range(layers):
                x, attn_weight = self.resblocks[i](x)
                attn_weights.append(attn_weight)
        return x, attn_weights


class CustomVisionTransformer(nn.Module):
    """A customized VisionTransformer to support CAM calculation."""

    def __init__(self, model: VisionTransformer):
        """Initialize the wrapper.

        Args:
            model (VisionTransformer): the VisionTransformer to be wrapped.

        """
        super().__init__()
        for k, v in vars(model).items():
            setattr(self, k, v)
        self.patch_size = self.conv1.kernel_size[0]
        self.transformer = CustomTransformer(self.transformer)

    def forward(self, x: torch.Tensor, H, W):
        self.positional_embedding_new = upsample_position_embedding(
            self.positional_embedding,
            (H // self.patch_size, W // self.patch_size))
        # shape = [*, width, grid, grid]
        x = self.conv1(x)
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # shape = [*, grid ** 2, width]
        x = x.permute(0, 2, 1)
        zeros = torch.zeros(x.shape[0],
                            1,
                            x.shape[-1],
                            dtype=x.dtype,
                            device=x.device)
        # shape = [*, grid ** 2 + 1, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + zeros, x], dim=1)
        x = x + self.positional_embedding_new.to(x.dtype)
        x = self.ln_pre(x)
        # NLD -> LND
        x = x.permute(1, 0, 2)
        x, attn_weight = self.transformer(x)
        return x, attn_weight


class CLIPWrapper(nn.Module):
    """A wrapper for CLIP to support forward with a list of text inputs."""

    def __init__(self, clip_model: CLIP):
        """Initialize the wrapper.

        Args:
            clip_model (CLIP): the CLIP model to be wrapped.

        """
        super().__init__()
        # copy all attributes from clip_model to self
        for k, v in vars(clip_model).items():
            setattr(self, k, v)
        self.visual = CustomVisionTransformer(self.visual)
        self.transformer = CustomTransformer(self.transformer)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, H, W):
        return self.visual(image.type(self.dtype), H, W)

    def encode_text(self, text):
        x = self.token_embedding(text).type(
            self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding
        # (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)
              ] @ self.text_projection

        return x

    def pool_visual(self, x, use_cls_token=False):
        if use_cls_token:
            return x[:, 0]
        else:
            return torch.mean(x[:, 1:, :], dim=1)

    def forward_last_layer(
            self,
            image_features,
            text_features,
            use_cls_token=False,
            repeat_last=True):
        """Forward the last layer of CLIP.

        Args:
            image_features (torch.Tensor): the image features.
            text_features (torch.Tensor): the text features.
            use_cls_token (bool, optional): whether to use the CLS token.
                Defaults to False.
            repeat_last (bool, optional): whether to repeat the last layer.
                Defaults to True.

        Returns:
            torch.Tensor: the logits.
            torch.Tensor: the attention weights.
        """
        if repeat_last:
            x, attention_weight = self.visual.transformer.resblocks[
                self.visual.transformer.layers-1](image_features)
        else:
            x = image_features
            attention_weight = None
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x)
        x = self.pool_visual(x, use_cls_token=use_cls_token)

        if self.visual.proj is not None:
            x = x @ self.visual.proj

        image_features = x

        # normalized features
        image_features = image_features / image_features.norm(
            dim=1, keepdim=True)
        text_features = text_features / text_features.norm(
            dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        # shape = [global_batch_size, global_batch_size]
        logits_per_image = F.softmax(logits_per_image.float(), dim=-1)

        return logits_per_image, attention_weight

    def forward(self, image, text, H=224, W=224):
        with torch.no_grad():
            text_features = self.encode_text(text)
            feature_map, _ = self.visual(image.type(self.dtype), H, W)

            logits_per_image, _ = self.forward_last_layer(
                feature_map,
                text_features,
                use_cls_token=True,
                repeat_last=False)
        return logits_per_image
