"""load clip

The code below is based on https://github.com/rom1504/clip-retrieval/
"""
from functools import lru_cache
from torch import autocast, nn
import torch
import clip
from PIL import Image
import time


class OpenClipWrapper(nn.Module):
    """
    Wrap OpenClip for managing input types
    """

    def __init__(self, inner_model, device):
        super().__init__()
        self.inner_model = inner_model
        self.device = torch.device(device=device)
        if self.device.type == "cpu":
            self.dtype = torch.float32
        else:
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def encode_image(self, image):
        if self.device.type == "cpu":
            return self.inner_model.encode_image(image)
        with autocast(device_type=self.device.type, dtype=self.dtype):
            return self.inner_model.encode_image(image)

    def encode_text(self, text):
        if self.device.type == "cpu":
            return self.inner_model.encode_text(text)
        with autocast(device_type=self.device.type, dtype=self.dtype):
            return self.inner_model.encode_text(text)

    def forward(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)


def load_open_clip(clip_model, use_jit=True, device="cuda", clip_cache_path=None):
    """load open clip"""

    import open_clip  # pylint: disable=import-outside-toplevel

    torch.backends.cuda.matmul.allow_tf32 = True

    pretrained = dict(open_clip.list_pretrained())
    checkpoint = pretrained[clip_model]
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model, pretrained=checkpoint, device=device, jit=use_jit, cache_dir=clip_cache_path
    )
    model = OpenClipWrapper(inner_model=model, device=device)
    model.to(device=device)
    return model, preprocess


@lru_cache(maxsize=None)
def get_tokenizer(clip_model):
    """Load clip"""
    if clip_model.startswith("open_clip:"):
        import open_clip  # pylint: disable=import-outside-toplevel

        clip_model = clip_model[len("open_clip:") :]
        return open_clip.get_tokenizer(clip_model)
    else:
        return lambda t: clip.tokenize(t, truncate=True)


@lru_cache(maxsize=None)
def load_clip_without_warmup(clip_model, use_jit, device, clip_cache_path):
    """Load clip"""
    if clip_model.startswith("open_clip:"):
        clip_model = clip_model[len("open_clip:") :]
        model, preprocess = load_open_clip(clip_model, use_jit, device, clip_cache_path)
    else:
        model, preprocess = clip.load(clip_model, device=device, jit=use_jit, download_root=clip_cache_path)
    return model, preprocess


@lru_cache(maxsize=None)
def load_clip(clip_model="ViT-B/32", use_jit=True, warmup_batch_size=1, clip_cache_path=None, device=None):
    """Load clip then warmup"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip_without_warmup(clip_model, use_jit, device, clip_cache_path)

    start = time.time()
    print(f"warming up with batch size {warmup_batch_size} on {device}", flush=True)
    warmup(warmup_batch_size, device, preprocess, model)
    duration = time.time() - start
    print(f"done warming up in {duration}s", flush=True)
    return model, preprocess


def warmup(batch_size, device, preprocess, model):
    fake_img = Image.new("RGB", (224, 224), color="red")
    fake_text = ["fake"] * batch_size
    image_tensor = torch.cat([torch.unsqueeze(preprocess(fake_img), 0)] * batch_size).to(device)
    text_tokens = clip.tokenize(fake_text).to(device)
    for _ in range(2):
        with torch.no_grad():
            model.encode_image(image_tensor)
            model.encode_text(text_tokens)