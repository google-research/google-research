'''
@File       :   utils.py
@Time       :   2023/04/05 19:18:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
* Based on CLIP code base
* https://github.com/openai/CLIP
* Checkpoint of CLIP/BLIP/Aesthetic are from:
* https://github.com/openai/CLIP
* https://github.com/salesforce/BLIP
* https://github.com/christophschuhmann/improved-aesthetic-predictor
'''

import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from .ImageReward import ImageReward
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from .models.CLIPScore import CLIPScore
from .models.BLIPScore import BLIPScore
from .models.AestheticScore import AestheticScore

_MODELS = {
    "ImageReward-v1.0": "https://huggingface.co/THUDM/ImageReward/blob/main/ImageReward.pt",
}


def available_models() -> List[str]:
    """Returns the names of available ImageReward models"""
    return list(_MODELS.keys())


def ImageReward_download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)
    hf_hub_download(repo_id="THUDM/ImageReward", filename=filename, local_dir=root)
    return download_target


def load(name: str = "ImageReward-v1.0", device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None, med_config: str = None):
    """Load a ImageReward model

    Parameters
    ----------
    name : str
        A model name listed by `ImageReward.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    download_root: str
        path to download the model files; by default, it uses "~/.cache/ImageReward"

    Returns
    -------
    model : torch.nn.Module
        The ImageReward model
    """
    if name in _MODELS:
        model_path = ImageReward_download(_MODELS[name], download_root or os.path.expanduser("~/.cache/ImageReward"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    print('load checkpoint from %s'%model_path)
    state_dict = torch.load(model_path, map_location='cpu')
    
    # med_config
    if med_config is None:
        med_config = ImageReward_download("https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json", download_root or os.path.expanduser("~/.cache/ImageReward"))
    
    model = ImageReward(device=device, med_config=med_config).to(device)
    msg = model.load_state_dict(state_dict,strict=False)
    print("checkpoint loaded")
    model.eval()

    return model


_SCORES = {
    "CLIP": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "BLIP": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth",
    "Aesthetic": "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth",
}


def available_scores() -> List[str]:
    """Returns the names of available ImageReward scores"""
    return list(_SCORES.keys())


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


def load_score(name: str = "CLIP", device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None):
    """Load a ImageReward model

    Parameters
    ----------
    name : str
        A model name listed by `ImageReward.available_models()`

    device : Union[str, torch.device]
        The device to put the loaded model

    download_root: str
        path to download the model files; by default, it uses "~/.cache/ImageReward"

    Returns
    -------
    model : torch.nn.Module
        The ImageReward model
    """
    model_download_root = download_root or os.path.expanduser("~/.cache/ImageReward")
    
    if name in _SCORES:
        model_path = _download(_SCORES[name], model_download_root)
    else:
        raise RuntimeError(f"Score {name} not found; available scores = {available_scores()}")

    print('load checkpoint from %s'%model_path)
    if name == "BLIP":
        state_dict = torch.load(model_path, map_location='cpu')
        med_config = ImageReward_download("https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json", model_download_root)
        model = BLIPScore(med_config=med_config, device=device).to(device)
        model.blip.load_state_dict(state_dict['model'],strict=False)
    elif name == "CLIP":
        model = CLIPScore(download_root=model_download_root, device=device).to(device)
    elif name == "Aesthetic":
        state_dict = torch.load(model_path, map_location='cpu')
        model = AestheticScore(download_root=model_download_root, device=device).to(device)
        model.mlp.load_state_dict(state_dict,strict=False)
    else:
        raise RuntimeError(f"Score {name} not found; available scores = {available_scores()}")
    
    print("checkpoint loaded")
    model.eval()

    return model
