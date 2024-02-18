import torch


def IoU(mask1, mask2, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) between prediction and
    ground truth masks.
    Args:
        mask1: A torch.Tensor denoting the prediction, shape (N, H, W),
        where N is the number of masks.
        mask2: A torch.Tensor denoting the ground truth, shape (N, H, W),
        where N is the number of masks.
    """
    if threshold > 0:
        mask1, mask2 = (mask1 > threshold).to(torch.bool), (
            mask2 > threshold
        ).to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    if union.sum() == 0:
        return 0
    return (intersection.to(torch.float) / union).mean().item()


def IoM(pred, target, min_pred_threshold=0.2):
    """
    Calculate Intersection over the area of gt Mask and pred Mask (IoM)
    between prediction and each ground truth masks.
    Precaution:
        this function works for prediction and target that are binary masks,
        where 1 represents the mask and 0 represents the background.
    Args:
        pred: A torch.Tensor denoting the prediction, shape (N, H, W),
        where N is the number of masks.
        target: A torch.Tensor denoting the ground truth, shape (N, H, W),
        where N is the number of masks.
    Return:
        ious: A torch.Tensor denoting the IoU, shape (N,).
    """
    # calculate the intersection over all masks
    intersection = torch.einsum("mij,nij->mn", pred.to(target.device), target)
    area_pred = torch.einsum("mij->m", pred)
    area_target = torch.einsum("nij->n", target)
    # we calculate the IoM by dividing the intersection over the minimum area.
    iom_target = torch.einsum("mn,n->mn", intersection, 1 / area_target)
    # iom_target = intersection / area_target
    iom_pred = torch.einsum("mn,m->mn", intersection, 1 / area_pred)
    # iom_pred = intersection / area_pred
    # iom = torch.max(iom_target, iom_pred)
    # if the intersection is smaller than a certain percentage of the area of
    # the pred mask, we consider it as background.
    iom_target[iom_pred < min_pred_threshold] = 0
    # we consider the IoM as the maximum IoM between the pred mask and
    # the target mask.
    iom = torch.max(iom_target, iom_pred)
    iom = iom.max(dim=0)[0]
    return iom
