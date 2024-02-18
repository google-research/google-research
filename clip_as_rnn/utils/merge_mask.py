import torch
import torch.nn.functional as F
import numpy as np

import cv2


IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512

# from segment_anything import SamPredictor, sam_model_registry,
# SamAutomaticMaskGenerator
# GroundedSAM
# from GroundedSAM import get_grounding_output, load_dino_image,
# load_dino_model

# def inference_gsam(cfg, model, image_path, text_prompt, predictor, device):
#     # run grounding dino model
#     image_pil, image = load_dino_image(image_path)
#     boxes_filt, pred_phrases = get_grounding_output(
#         model, image, text_prompt, cfg.groundeddino_model.box_threshold,
#         cfg.groundeddino_model.text_threshold, device=device
#     )
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     predictor.set_image(image)
#     size = image_pil.size
#     H, W = size[1], size[0]
#     for i in range(boxes_filt.size(0)):
#         boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
#         boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
#         boxes_filt[i][2:] += boxes_filt[i][:2]

#     boxes_filt = boxes_filt.cpu()
#     transformed_boxes = predictor.transform.apply_boxes_torch(
#         boxes_filt, image.shape[:2]).to(device)
#     if transformed_boxes.size()[0] == 0:
#         return np.full((1, H, W), False), np.array([0.0])
#     masks, scores, _ = predictor.predict_torch(
#         point_coords=None,
#         point_labels=None,
#         boxes=transformed_boxes.to(device),
#         multimask_output=False,
#     )

#     masks, scores = masks.cpu().numpy(), scores.cpu().numpy()
#     masks, scores = np.squeeze(masks, axis=1), np.squeeze(scores, axis=1)
#     # print(type(masks[0]), type(scores[0]))
#     # print(masks.shape, scores.shape)
#     return masks, scores


def merge_masks_simple(all_masks,
                       target_h,
                       target_w,
                       threshold=0.5,
                       scores=None):
    if scores is not None:
        merged_mask = torch.sum(all_masks * scores[:, None, None], dim=0)
        merged_mask /= torch.sum(scores)
    merged_mask = merged_mask.detach().cpu().numpy()
    # resize the mask to the target size
    merged_mask = cv2.resize(merged_mask, (target_w, target_h))
    merged_mask = np.where(
        merged_mask >= threshold, 1, 0).astype(np.uint8)
    if np.sum(merged_mask) <= 0.05 * (target_h * target_w):
        merged_mask = torch.any(all_masks > 0, dim=0)
        merged_mask = merged_mask.detach().cpu().numpy().astype(np.uint8)
        # resize the mask to the target size
        merged_mask = cv2.resize(merged_mask, (target_w, target_h))
        merged_mask = merged_mask > threshold
    merged_mask = torch.from_numpy(merged_mask).float()
    return merged_mask[None]


def merge_masks(all_masks, target_h, target_w, threshold=0.5):
    all_masks = torch.from_numpy(np.stack(all_masks)).float()

    # merged_mask = torch.mean(all_masks, dim=0)
    # merged_mask = torch.where(merged_mask >= 0.5, 1, 0)
    # if torch.sum(merged_mask) <= 0.05 * (target_h * target_w):
    #     merged_mask = torch.any(all_masks.float(), dim=0)
    mask_tensor = F.interpolate(all_masks[None],
                                size=(target_h, target_w),
                                mode='bilinear').squeeze(0)
    bg_mask = threshold * torch.ones((1, target_h, target_w))
    merged_mask = torch.cat([bg_mask, mask_tensor], dim=0)
    mask_idx = torch.argmax(merged_mask, dim=0)
    merged_mask = mask_idx > 0
    if merged_mask.sum() <= 0.05 * (target_h * target_w):
        merged_mask = torch.any(mask_tensor, dim=0)
    return merged_mask.float()[None]
