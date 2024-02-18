import torch
import torch.nn.functional as F
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils


class DenseCRF(object):
    def __init__(
        self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std
    ):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std,
            srgb=self.bi_rgb_std,
            rgbim=image,
            compat=self.bi_w,
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q


class PostProcess:
    def __init__(self, device):
        self.device = device
        self.postprocessor = DenseCRF(
            iter_max=10,
            pos_xy_std=1,
            pos_w=3,
            bi_xy_std=67,
            bi_rgb_std=3,
            bi_w=4,
        )

    def apply_crf(self, image, cams, bg_factor=1.0):
        bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), bg_factor)
        cams = np.concatenate((bg_score, cams), axis=0)
        prob = cams

        image = image.astype(np.uint8).transpose(1, 2, 0)
        prob = self.postprocessor(image, prob)

        label = np.argmax(prob, axis=0)

        label_tensor = torch.from_numpy(label).long()
        refined_mask = F.one_hot(label_tensor).to(device=self.device)
        refined_mask = refined_mask.permute(2, 0, 1)
        refined_mask = refined_mask[1:].float()
        return refined_mask

    def __call__(self, image, cams, separate=False, bg_factor=1.0):
        mean_bgr = (104.008, 116.669, 122.675)
        # covert Image to numpy array
        image = np.array(image).astype(np.float32)

        # RGB -> BGR
        image = image[:, :, ::-1]
        # Mean subtraction
        image -= mean_bgr
        # HWC -> CHW
        image = image.transpose(2, 0, 1)

        if isinstance(cams, torch.Tensor):
            cams = cams.cpu().detach().numpy()
        if separate:
            refined_mask = [
                self.apply_crf(image, cam[None], bg_factor) for cam in cams
            ]
            refined_mask = torch.cat(refined_mask, dim=0)
        else:
            refined_mask = self.apply_crf(image, cams, bg_factor)

        return refined_mask
