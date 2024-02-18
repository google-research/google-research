from . import dino
from . import dino2


def dino_factory(dino_arch, pretrained_path=None):
    if dino_arch == 'base_v1':
        patch_size = 8
        url = ('https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/'
               'dino_vitbase8_pretrain.pth')
        path = url if pretrained_path is None else pretrained_path
        feat_dim = 768

        return dino.ViTFeat(path, feat_dim, 'base', 'k', patch_size)

    elif dino_arch == 'base_v2':
        url = ('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/'
               'dinov2_vitb14_pretrain.pth')
        path = url if pretrained_path is None else pretrained_path
        patch_size = 14
        feat_dim = 768
        return dino2.ViTFeat(path, feat_dim, 'base', 'k', patch_size)

    elif dino_arch == 'large_v2':
        url = ('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/'
               'dinov2_vitl14_pretrain.pth')
        path = url if pretrained_path is None else pretrained_path
        patch_size = 14
        feat_dim = 1024
        return dino2.ViTFeat(path, feat_dim, 'large', 'k', patch_size)
