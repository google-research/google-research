# You Look Twice: GaterNet for Dynamic Filter Selection in CNNs
This repository contains the model and evaluation code for ResNet-20-Gated and ResNet-56-Gated on CIFAR-10 as in the following paper. \
[You Look Twice: GaterNet for Dynamic Filter Selection in CNNs](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_You_Look_Twice_GaterNet_for_Dynamic_Filter_Selection_in_CNNs_CVPR_2019_paper.pdf) \
Zhourong Chen, Yang Li, Samy Bengio, Si Si. \
The paper is accepted to IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

The requirement list in requirements.txt is for reproducing the results of the paper. You don't have to  strictly follow the list if you just want to run the code instead of reproducing the exact results.

Please download our checkpoints by manually copying this link to your web browser: http://storage.googleapis.com/gresearch/gaternet/GaterNet_Checkpoints.tar.gz

## Evaluate ResNet-20-Gated on CIFAR-10.
```bash
python main.py \
--backbone-depth 20 \
--checkpoint-file /path/to/downloaded/checkpoints/ResNet-20-Gated.pth \
--data-dir /path/to/download/data/
```

## Evaluate ResNet-56-Gated on CIFAR-10.
```bash
python main.py \
--backbone-depth 56 \
--checkpoint-file /path/to/downloaded/checkpoints/ResNet-56-Gated.pth \
-data-dir /path/to/download/data/
```

If you find the paper or code useful, please cite our paper:

```
@inproceedings{gaternet,
  title={You look twice: Gaternet for dynamic filter selection in cnns},
  author={Chen, Zhourong and Li, Yang and Bengio, Samy and Si, Si},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9172--9180},
  year={2019}
}
```
