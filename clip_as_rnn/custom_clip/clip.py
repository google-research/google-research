import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List, Optional
from pkg_resources import packaging

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model import build_model
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils.nlp import load_spacy_model, process_sentence

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "tokenize", "forward_clip"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(
            _MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(
                opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(
                    f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones(
        []).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes(
        "prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(
            lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] +
                  _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def forward_text(text_list, model, device, return_text_attention):
    """
    Forward loop for CLIP model.
    Args:
        model: An instance of CLIP model.
        text: A str as the text input.
        device: A torch.device object.
        return_text_attention: A bool indicating whether to return the attention map.
    """
    text = tokenize(text_list).to(device)
    text_features = model.encode_text(text)
    return text_features / text_features.norm(dim=-1, keepdim=True)


def forward_clip(image,
                 text,
                 model,
                 return_text_attention=False,
                 return_logits=False,
                 phrase_mix_alpha=0.,
                 batch_forward=False):
    """
    Forward loop for CLIP model.
    Args:
        image: A np.ndarray for the input image.
        text: A str or a list of str as the text input.
        model: An instance of CLIP model.
        return_attention: A bool indicating whether to return the attention map.
        return_text_attention: A bool indicating whether to return the attention map.
        return_logits: A bool indicating whether to return the logits.
        phrase_mix_alpha: A float indicating the mixing ratio of the phrase.
        image_mask: A np.ndarray for the input image mask.
        batch_forward: A bool indicating whether to forward in batch.
    Returns:
        logits_per_image: A torch.Tensor of shape [1, num_texts] as the logits.
        feature_maps: A list of torch.Tensor as the feature maps.
        attn_map: A torch.Tensor of shape [num_layers, 1, height, width] as the attention map.
    """

    if isinstance(text, str):
        text_list = [text]
    else:
        text_list = text
    # with torch.no_grad():
    feature_dicts = model.encode_image(image)
    if isinstance(feature_dicts, dict):
        image_features = feature_dicts['output']
        feature_maps = feature_dicts['tokens']
        attn_map = feature_dicts['last_attention']
    else:
        image_features = feature_dicts
        feature_maps = None
        attn_map = None

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp()

    text_features = forward_text(
        text_list, model, image_features.device, return_text_attention)
    if batch_forward:
        # we do not support multi-text for batch_forward
        logits_per_image = logit_scale * torch.einsum('bd, bd -> b',
                                                      image_features,
                                                      text_features)[..., None]
    else:
        logits_per_image = logit_scale * image_features @ text_features.T

    if phrase_mix_alpha > 0. and isinstance(text, str):
        nlp = load_spacy_model()
        phrase = process_sentence(text, nlp)
        phrase_features = forward_text(
            phrase, model, image_features.device, return_text_attention)
        if batch_forward:
            # we do not support multi-text for batch_forward
            phrase_logits_per_image = logit_scale * torch.einsum('bd, bd -> b',
                                                                 image_features,
                                                                 phrase_features)[..., None]
        else:
            phrase_logits_per_image = logit_scale * image_features @ phrase_features.T
        logits_per_image = phrase_mix_alpha * phrase_logits_per_image + \
            (1-phrase_mix_alpha) * logits_per_image
        text_features = phrase_mix_alpha * phrase_features + \
            (1-phrase_mix_alpha) * text_features

    if return_logits:
        # For extract clip embedding
        return logits_per_image, feature_maps, attn_map, text_features
    text_prediction = (text_features * image_features)

    return text_prediction, feature_maps, attn_map, text_features


# def forward_clip_extract_embedding(image,
#                  text,
#                  model,
#                  return_attention=True,
#                  return_text_attention=False,
#                  return_logits=False,
#                  phrase_mix_alpha=0.,
#                  image_mask=None,
#                  require_emb=False):
#     """
#     Forward loop for CLIP model.
#     Args:
#         image: A np.ndarray for the input image.
#         text: A str or a list of str as the text input.
#         model: An instance of CLIP model.
#         return_attention: A bool indicating whether to return the attention map.
#         return_text_attention: A bool indicating whether to return the attention map.
#         return_logits: A bool indicating whether to return the logits.
#         phrase_mix_alpha: A float indicating the mixing ratio of the phrase.
#         image_mask: A np.ndarray for the input image mask.

#     Returns:
#         logits_per_image: A torch.Tensor of shape [1, num_texts] as the logits.
#         feature_maps: A list of torch.Tensor as the feature maps.
#         attn_map: A torch.Tensor of shape [num_layers, 1, height, width] as the attention map.
#     """
#     assert return_logits and require_emb

#     if isinstance(text, str):
#         text_list = [text]
#     else:
#         text_list = text
#     # with torch.no_grad():
#     feature_dicts = model.encode_image(image, return_attention=return_attention, image_mask=image_mask)
#     image_features = feature_dicts['output']
#     feature_maps = feature_dicts['tokens']
#     attn_map = feature_dicts['last_attention']


#     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#     logit_scale = model.logit_scale.exp()

#     text_features = forward_text(text_list, model, image_features.device, return_text_attention)
#     logits_per_image = logit_scale * image_features @ text_features.T

#     if phrase_mix_alpha > 0. and isinstance(text, str):
#         nlp = load_spacy_model()
#         phrase = process_sentence(text, nlp)
#         phrase_features = forward_text(phrase, model, image_features.device, return_text_attention)
#         phrase_logits_per_image = logit_scale * image_features @ phrase_features.T
#         logits_per_image = phrase_mix_alpha * phrase_logits_per_image + (1-phrase_mix_alpha) * logits_per_image
#         text_features = phrase_mix_alpha * phrase_features + (1-phrase_mix_alpha) * text_features


#     if return_logits:
#         return logits_per_image, feature_maps, attn_map

#     text_prediction = (text_features * image_features)

#     # For extract clip embedding
#     if require_emb:
#         return text_prediction, feature_maps, attn_map, text_features, image_features

#     return text_prediction, feature_maps, attn_map
