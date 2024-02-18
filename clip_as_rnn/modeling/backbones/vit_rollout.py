from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from custom_clip import clip
from custom_clip.transformer import VisionTransformer
from utils.utils import normalize
from . import dino_factory


@torch.no_grad()
def attn_denoise(text_img_clues, patch_attn):
    """
    Gradcam algorithm for visualizing input saliency for any backbones.
    Args:
        text_img_clues: A tensor of attention gradients from the model.
            Shape: [num_classes, side_length, side_length]
        patch_attn: A tensor list of attention maps from the model or DINO.
            Shape: [1, 1, num_patches, num_patches]
    """
    num_texts, h, w = text_img_clues.shape
    cls_weights = text_img_clues.flatten(1)
    # cls_weights = cls_weights[:, None, :]
    cls_weights = cls_weights.view(num_texts, 1, h, w)
    patch_side = int(patch_attn.shape[0] ** 0.5)
    cls_weights = F.interpolate(cls_weights, size=(
        patch_side, patch_side), mode='bicubic')
    cls_weights = cls_weights.flatten(1)

    # attn-based:
    # cls_weights = F.normalize(cls_weights, dim=-1, p=2)
    # patch_attn = F.normalize(patch_attn, dim=0, p=2)
    norm_attn = patch_attn / patch_attn.sum(dim=0, keepdim=True)
    norm_attn = norm_attn / norm_attn.sum(dim=1, keepdim=True)
    norm_attn = (norm_attn + norm_attn.transpose(1, 0)) / 2
    norm_attn = norm_attn @ norm_attn
    patch_attn = norm_attn * patch_attn
    mask = cls_weights @ patch_attn / patch_attn.sum(dim=0)
    # mask = cls_weights * mask
    mask = mask.view(num_texts, patch_side, patch_side)
    return mask
    # element-wise:
    # cls_weights = cls_weights[:, None, :]
    # we use the attention weights as a way to weight the gradients
    # attn_mask = (patch_attn[None] * cls_weights).sum(dim=-1)
    # mask = cls_weights.squeeze(1) * attn_mask
    # mask = mask.view(num_texts, patch_side, patch_side)
    # return mask

    # softmax-based thresholding:
    # cls_weights = F.normalize(cls_weights, dim=-1, p=2)
    # patch_attn = F.normalize(patch_attn, dim=0, p=2)
    # # mask = cls_weights @ patch_attn / patch_attn.sum(dim=0)
    # mask = cls_weights @ patch_attn * 100 # 100.0 is a temperature
    # mask = F.softmax(mask, dim=-1)
    # val, idx = torch.sort(mask, dim=-1)
    # val /= torch.sum(val, dim=-1, keepdim=True)
    # cumval = torch.cumsum(val, dim=-1)
    # th_attn = cumval > (1 - keep_threshold)
    # idx2 = torch.argsort(idx, dim=-1)
    # th_attn = th_attn.gather(dim=-1, index=idx2)
    # return th_attn.view(num_texts, patch_side, patch_side).float()


@torch.no_grad()
def grad_rollout(patch_attns,
                 gradients,
                 device: torch.device = "cuda",
                 is_vit=False):
    """
    Gradient rollout algorithm for visualizing input saliency for
    VisionTransformer.
    NOTE: here we only support batch_size = 1.
    Args:
        patch_attns: A list of attention maps from the model. Shape:
        [batch_size * num_heads, num_patches, num_patches]
        gradients: A tensor list of attention gradients from the model. Shape:
            [num_classes, batch_size, num_heads, num_patches, num_patches]
        discard_ratio: A float for the ratio of attention to discard.
        thres: A float for the threshold to apply to the attention map.
        device: A torch.device for the device to use.
        layers_to_inc: An int for the number of layers to include in the
            rollout algorithm.
    Returns:
        A torch.Tensor representing the saliency map with shape
            (batch_size, num_classes, image_size, image_size).
    """
    num_classes, _, _, _, ntokens = gradients[0].shape
    ntokens = ntokens - 1 if is_vit else ntokens
    side_len = int(ntokens ** 0.5)
    result = torch.zeros(num_classes, ntokens, device=device)
    for _, (patch_attn, grad) in enumerate(zip(patch_attns, gradients)):
        patch_attn = patch_attn.to(device)
        alpha = grad.clamp(min=0).mean(dim=[-1, -2], keepdim=True)
        # gradCAM
        weights = torch.sum(grad * alpha, dim=2).clamp(min=0)
        if len(patch_attn.shape) == 3:  # ViT patch_attn
            text_img_clues = weights[..., 0, 1:]
            patch_attn = patch_attn[:, 1:, 1:].mean(dim=0)
        elif is_vit:  # ViT & DINO
            text_img_clues = weights[..., 0, 1:]
        else:  # ConvNeXt & DINO
            text_img_clues = weights.flatten(2)
        norm_cam = normalize(text_img_clues, dim=-1)
        text_img_clues[norm_cam < 0.1] = 0.
        # norm_cam = norm_cam.view(num_classes, side_len, side_len)
        text_img_clues = text_img_clues.view(num_classes, side_len, side_len)
        # binary_masks = filter_masks(norm_cam,
        #                             mask_threshold=0.3,
        #                             min_area_ratio=0.,
        #                             return_instances=False,
        #                             device=device)
        # binary_masks = torch.cat(binary_masks, dim=0)
        mask = attn_denoise(text_img_clues, patch_attn)
        result = mask.flatten(1)
    result = normalize(result, dim=-1)
    side_len = int(result.shape[-1] ** 0.5)
    result = result.view(num_classes, side_len, side_len)
    confidence = (result).sum(dim=[1, 2])  # text_img_clues
    return result.to(device), confidence.to(device)


class CLIPAttentionGradRollout:
    """
    Rollout class the attention gradients for CLIP.
    """

    def __init__(
        self,
        model,
        attention_layer_name="attn.dropout",
        key_layer_name="47.attn.k_idt",
        discard_ratio=0.5,
        num_layers=1,
        dino_arch=None,
        dino_path=None,
        dino_device='cpu',
    ):
        """
        Rollout class the attention gradients for CLIP.

        Args:
            model: A nn.Module representing the CLIP model.
            attention_layer_name: A str for the name of the attention layer to
                rollout.
            discard_ratio: A float for the ratio of attention to discard.
                Not used by default.
        """
        self.model = model
        self.discard_ratio = discard_ratio
        self.num_layers = num_layers
        self.module_list = []
        self.dino_device = dino_device
        for name, module in self.model.named_modules():
            if attention_layer_name in name and "visual" in name:
                module.register_forward_hook(self.get_attention)
        for name, module in self.model.named_modules():
            if key_layer_name in name and "visual" in name:
                module.register_forward_hook(self.get_attn_key)

        self.dino_model = None
        if dino_arch is not None:
            self.dino_model = dino_factory(dino_arch, dino_path)
            self.dino_model = self.dino_model.to(dino_device)

        self.is_vit = False
        if 'attn' in attention_layer_name:
            self.is_vit = True
        self.attentions = []
        self.attention_gradients = []
        self.feats = []

    def get_attention(self, module, input, output):
        """
        Forward hook to get the attention maps from the CLIP model.
        """
        self.attentions.append(output)

    def get_attn_key(self, module, input, output):
        """
        Forward hook to get the key feature maps of the attention blocks
            from the CLIP model.
        """
        self.feats.append(output)

    def get_attention_gradient(self, target):
        """
        Backward hook to get the attention gradients from the CLIP model.
        """
        def get_grad_single(t, attn):
            grads = []
            for cls_id in range(num_classes):
                self.model.zero_grad()
                # here torch.autograd.grad does not support bacth backward.
                # so we can only compute the gradient for each sample
                # in the batch.
                grad = torch.autograd.grad(t[..., cls_id],
                                           [attn.requires_grad_(True)],
                                           retain_graph=True)[0]
                grads.append(grad.detach())
            return torch.stack(grads, dim=0)

        grad_list = []
        batch_size, num_classes = target.shape
        for attn in self.attentions[-self.num_layers:]:
            n_tokens = attn.size(-1) if self.is_vit else attn.size(-2)
            batch_grads = get_grad_single(target, attn)
            if self.is_vit:
                grad_list.append(batch_grads.view(
                    num_classes, batch_size, -1, n_tokens, n_tokens))
            else:
                grad_list.append(batch_grads.permute(
                    0, 1, 4, 2, 3).contiguous())
        return grad_list

    def load_dino(self, dino_image, clip_h, clip_w):
        """
        Load the DINO model and get the attention maps.
        """
        # dino_image = DINO_transform(dino_image)[None]
        # dino_image = dino_image.to(self.dino_device)
        dino_feat = self.dino_model(dino_image)[0]
        dino_feat = F.normalize(dino_feat, p=2, dim=0)
        # target_h, target_w = max(clip_h, dino_len), max(clip_w, dino_len)
        target_h, _ = clip_h, clip_w
        # if dino_len != target_h:
        #     dino_feat = dino_feat.view(-1, dino_dim, dino_len, dino_len)
        # dino_feat = F.interpolate(dino_feat,
        #                           size=(clip_h, clip_w),
        #                           mode='bicubic')
        if clip_h != target_h:
            pass  # TODO: interpolate CLIP feature
        # dino_feat = dino_feat.view(dino_dim, target_h * target_w)
        dino_attn = (dino_feat.transpose(0, 1) @ dino_feat).detach()
        dino_attn = [F.softmax(dino_attn, dim=-1)]
        return dino_attn

    def __call__(
            self,
            image,
            text,
            phrase_mix_alpha=0.0,
            score_map_ratio=0.0,
            image_dino=None,
            return_key_features=False,
            return_confidence=False,
            return_text_prediction=False,):
        self.model.eval()
        self.attentions.clear()
        self.attention_gradients.clear()
        self.feats.clear()
        self.model.zero_grad()
        # self.model.visual.trunk.set_grad_checkpointing(True)
        if return_text_prediction:
            if isinstance(text[0], list):
                for template_idx, template in enumerate(text):
                    text_prediction, _, _, _ = clip.forward_clip(
                        image,
                        template,
                        self.model,
                        return_text_attention=False,
                        return_logits=True,
                        phrase_mix_alpha=phrase_mix_alpha,
                        # image_mask=image_mask
                    )
                    if template_idx == 0:
                        text_prediction_sum = text_prediction.detach().cpu()
                    else:
                        text_prediction_sum += text_prediction.detach().cpu()
            # apply softmax to get the text prediction
            else:
                text_prediction_sum, _, _, _ = clip.forward_clip(
                    image,
                    text,
                    self.model,
                    return_text_attention=False,
                    return_logits=True,
                    phrase_mix_alpha=phrase_mix_alpha,
                    # image_mask=image_mask
                )
            text_prediction = F.softmax(text_prediction_sum, dim=-1)
            return text_prediction
        else:
            text_prediction, feature_map, _, text_features = clip.forward_clip(
                image,
                text,
                self.model,
                return_text_attention=False,
                return_logits=True,
                phrase_mix_alpha=phrase_mix_alpha,
                # image_mask=image_mask
            )
        if score_map_ratio > 0:
            feature_map = feature_map / feature_map.norm(dim=-1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()
            score_map = logit_scale * feature_map @ text_features.T

        text_prediction.retain_grad()
        device = text_prediction.device
        self.attention_gradients = self.get_attention_gradient(text_prediction)
        dino_attn = None
        if self.dino_model is not None:
            if isinstance(image_dino, np.ndarray):
                image_dino = Image.fromarray(image_dino)

            if len(self.attentions[-1].shape) == 4:  # CNN
                clip_h, clip_w = self.attention_gradients[-1].shape[-2:]
            else:  # ViT
                clip_h = clip_w = int(feature_map.shape[1] ** 0.5)
            dino_attn = self.load_dino(image_dino, clip_h, clip_w)

        mask, confidence = grad_rollout(
            torch.stack(self.attentions[-self.num_layers:],
                        dim=0) if dino_attn is None else dino_attn,
            self.attention_gradients,
            device=device,
            is_vit=isinstance(self.model.visual, VisionTransformer),
        )
        if score_map_ratio > 0:
            mask = score_map * score_map_ratio + mask * (1 - score_map_ratio)

        results = {}
        results['cam_map'] = mask
        if return_key_features:
            results['key_feat'] = self.feats[-1]
        if return_confidence:
            results['confidence'] = confidence
        return results
