import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from .utils import *


class SAMPipeline:
    def __init__(self, checkpoint, model_type, device="cuda:0", points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        box_nms_thresh=0.7):
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.device = device
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.sam.to(device=self.device)
        self.load_mask_generator(points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh, box_nms_thresh=box_nms_thresh)


        # Default Prompt Args
        self.click_args = {"k": 5, "order": "max", "how_filter": "median"}
        self.box_args = None

    def load_sam(self):
        print("Loading SAM")
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        print("Loading Done")

    def load_mask_generator(self, points_per_side, pred_iou_thresh, stability_score_thresh, box_nms_thresh):
        print("Loading SAM")
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side= points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
        )
        print("Loading Done")

    # segment single object
    def segment_image_single(
        self,
        image_path,
        input_point=None,
        input_label=None,
        input_box=None,
        input_mask=None,
        multimask_output=True,
        visualize=False,
        save_path=None,
        fname="",
        image=None,
    ):
        if image is None:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_mask_r = np.resize(input_mask, (1, 256, 256))
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            mask_input=None,
            multimask_output=multimask_output,
        )

        if visualize:
            self.visualize(
                image,
                masks,
                scores,
                save_path,
                input_point=input_point,
                input_label=input_label,
                input_box=input_box,
                input_mask=input_mask,
                fname=fname,
            )

        return masks, scores, logits

    def segment_automask(
        self,
        image_path,
        visualize=False,
        save_path=None,
        image=None,
        fname="automask.jpg",
    ):
        if image is None:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_list, bbox_list = [], []
        masks = self.mask_generator.generate(image)
        mask_list.extend([mask["segmentation"] for mask in masks])
        bbox_list.extend([mask["bbox"] for mask in masks])

        if visualize:
            self.visualize_automask(image, masks, save_path, fname=fname)

        masks_arr, bbox_arr = np.array(mask_list), np.array(bbox_list)
        return masks_arr, bbox_arr, masks

    def visualize_automask(self, image, masks, save_path, fname=f"mask.jpg"):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis("off")
        plt.savefig(os.path.join(save_path, fname))

    def visualize(
        self,
        image,
        masks,
        scores,
        save_path,
        input_point=None,
        input_label=None,
        input_box=None,
        input_mask=None,
        fname="",
    ):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            if input_point is not None:
                show_points(input_point, input_label, plt.gca())
            if input_box is not None:
                show_box(input_box, plt.gca())
            if input_mask is not None:
                show_mask(input_mask[0], plt.gca(), True)
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis("off")
            plt.savefig(os.path.join(save_path, f"{fname}{i}.jpg"))

    # TODO: support batch, support mask
    def get_prompt(self, activation_map, prompt_types, click_args=None, box_args=None):
        (input_point, input_label, input_box, input_mask) = None, None, None, None
        for prompt_type in prompt_types:
            if prompt_type == "click":
                k, order, how_filter = (
                    click_args["k"],
                    click_args["order"],
                    click_args["how_filter"],
                )
                point_label = CAM2SAMClick(activation_map, k, order, how_filter)
                input_point, input_label = np.array(point_label[0]), np.array(
                    [1] * point_label[0].shape[0]
                )

            elif prompt_type == "box":
                box_label = CAM2SAMBox(activation_map)
                input_box = np.array(box_label[0])

        return input_point, input_label, input_box, input_mask


if __name__ == "__main__":
    # Model Parameters
    # model_dir = ""
    sam_checkpoint = "/datasets/jianhaoy/Cache/SAM/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # Image and Prompts
    image_path = "/homes/55/jianhaoy/projects/test/sample/dog1.jpg"
    input_point = np.array([[310, 310], [270, 270], [250, 250]])
    input_label = np.array([1, 1, 1])
    input_box = np.array([425, 600, 700, 875])
    save_path = "/homes/55/jianhaoy/projects/test/sample"
    multimask_output = True
    device = "cuda"

    pipeline = SAMPipeline(sam_checkpoint, model_type, device)
    pipeline.segment_image_single(
        image_path=image_path,
        input_point=input_point,
        input_label=input_label,
        input_box=input_box,
        input_mask=input_mask,
        multimask_output=multimask_output,
        visualize=True,
        save_path=save_path,
    )
