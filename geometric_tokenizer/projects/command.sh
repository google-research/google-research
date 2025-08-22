# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python tools/train.py ./projects/configs/myconfig_nusc/repro_minkunet/10sweeps/minkunetvfe_minkunethead_nusc_nosyncbn_sgd_highres.py --seed 0 --debug #--resume-from "/home/kevintw/code/trans3D/OccFormer/work_dirs/transfer/minkunetvfe_minkunethead_nusc_nosyncbn_sgd_idxmap/train_on_trainset/best_nuScenes_lidarseg_mean_epoch_24.pth"
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 python tools/train.py ./projects/configs/myconfig_nusc/repro_minkunet/minkunetvfe_minkunethead_nusc_nosyncbn_sgd_highres2048_64.py --seed 0 --debug #--resume-from "/home/kevintw/code/trans3D/OccFormer/work_dirs/transfer/minkunetvfe_minkunethead_nusc_nosyncbn_sgd_idxmap/train_on_trainset/best_nuScenes_lidarseg_mean_epoch_24.pth"
# CUDA_LAUNCH_BLOCKING=1 python tools/train.py ./projects/configs/myconfig_nusc/repro_minkunet/minkunetvfe_minkunethead_nusc_nosyncbn_sgd_highres2048_160.py --seed 0 --debug
# CUDA_LAUNCH_BLOCKING=1 python tools/train.py ./projects/configs/myconfig_nusc/repro_spunet/sparseunet_nusc_nosyncbn_adam_highres2048_160_dim4class16.py --seed 0 --debug
# CUDA_LAUNCH_BLOCKING=1 python tools/train.py ./projects/configs/myconfig_nusc/spunet_occformer/spunet_occformer_nusc_highres_rangeclip_freezespconv.py --seed 0 --debug
# CUDA_LAUNCH_BLOCKING=1 python tools/train.py ./projects/configs/myconfig_nusc/spunet_trans_encoder/spunet_transenc_nusc_highres_rangeclip_no-unet-decoder_class16_concat.py --seed 0 --debug --cfg-options optimizer.lr=0.0004
# CUDA_LAUNCH_BLOCKING=1 python tools/train.py ./projects/configs/myconfig_nusc/spunet_trans_encoder/spunet_transenc_nusc_highres_rangeclip_meanpool_class16_concat_shift_30ep.py --seed 0 --debug --cfg-options optimizer.lr=0.0004
# CUDA_LAUNCH_BLOCKING=1 python tools/train.py ./projects/configs/myconfig_nusc/pointpillar_occformer_nusc/pointpillarvfe_occformer_lowres_nusc_pointloader.py --seed 0 --debug --cfg-options optimizer.lr=0.0004
# CUDA_LAUNCH_BLOCKING=1 python tools/train.py ./projects/configs/myconfig_nusc/triplane_occformer_nusc/triplanevfe_occformer_lowres_nusc_pointloader_res32_pts256_nochannel.py --seed 0 --debug --cfg-options optimizer.lr=0.0004
CUDA_LAUNCH_BLOCKING=1 python tools/train.py ./projects/configs/myconfig_nusc/triplane_fpn_highres_nusc/triplanevfe_fpn_highres_nusc_linear_res16_pts256_nochannel.py --seed 0 --debug --cfg-options optimizer.lr=0.0004
# CUDA_LAUNCH_BLOCKING=1 python tools/train.py ./projects/configs/myconfig_nusc/repro_minkunet/minkunetvfe_minkunethead_nusc_nosyncbn_sgd_highres2048_160.py --seed 0 --debug
# CUDA_LAUNCH_BLOCKING=1 python tools/train.py ./projects/configs/myconfig_nusc/minkunetvfe_occformer_nusc.py --seed 0 --debug
# python tools/train.py ./projects/configs/myconfig_nusc/hardsimplevfe_occformer_nusc.py --seed 0 --debug
# python tools/train.py ./projects/configs/myconfig_nusc/pointpillarvfe_occformer_nusc.py --seed 0 --debug
# python tools/train.py ./projects/configs/occformer_nusc/occformer_nusc_r50_256x704_my_8layerdec.py --seed 0 --debug

# bash tools/dist_test.sh ./projects/configs/myconfig_nusc/triplane_fpn_highres_nusc/triplanevfe_fpn_highres_nusc_linear_res16_pts256_nochannel.py '/home/kevintw/code/trans3D/OccFormer/work_dirs/transfer/minkunetvfe_minkunethead_nusc_nosyncbn_sgd_idxmap/train_on_trainset/best_nuScenes_lidarseg_mean_epoch_24.pth' 8

# python tools/test.py ./projects/configs/myconfig_nusc/hardsimplevfe_occformer_nusc.py '/home/kevintw/code/trans3D/OccFormer/work_dirs/hardsimplevfe_occformer_nusc/train_on_trainset/latest.pth'  --seed 0 --eval 'segm'
# python tools/test.py ./projects/configs/myconfig_nusc/repro_minkunet/minkunetvfe_minkunethead_nusc_nosyncbn_sgd_highres2048_64.py '/home/kevintw/code/trans3D/OccFormer/work_dirs/transfer/minkunetvfe_minkunethead_nusc_nosyncbn_sgd_idxmap/train_on_trainset/best_nuScenes_lidarseg_mean_epoch_24.pth'  --seed 0 --eval 'segm'
# python tools/test.py ./projects/configs/myconfig_nusc/pointpillarvfe_occformer_nusc.py '/home/kevintw/code/trans3D/OccFormer/work_dirs/pointpillarvfe_occformer_nusc/train_on_trainset/best_nuScenes_lidarseg_mean_epoch_24.pth'  --seed 0 --eval 'segm'
# CUDA_VISIBLE_DEVICES=1 python tools/test.py ./projects/configs/myconfig_nusc/repro_minkunet/minkunetvfe_minkunethead_nusc_nosyncbn_sgd_highres64.py '/home/kevintw/code/trans3D/OccFormer/work_dirs/minkunetvfe_minkunethead_nusc_nosyncbn_sgd_highres64/train_on_trainset/best_nuScenes_lidarseg_mean_epoch_23.pth'  --seed 0 --eval 'segm'
# CUDA_VISIBLE_DEVICES=1 python tools/test.py ./projects/configs/myconfig_nusc/repro_minkunet/10sweeps/minkunetvfe_minkunethead_nusc_nosyncbn_sgd_highres.py '/home/kevintw/code/trans3D/OccFormer/work_dirs/minkunetvfe_minkunethead_nusc_nosyncbn_sgd_highres64/train_on_trainset/best_nuScenes_lidarseg_mean_epoch_23.pth'  --seed 0 --eval 'segm'
# CUDA_VISIBLE_DEVICES=1 python tools/test.py ./projects/configs/myconfig_nusc/triplane_fpn_highres_nusc/triplanevfe_fpn_highres_nusc_linear_res16_pts256_nochannel.py '/home/kevintw/code/trans3D/OccFormer/work_dirs/minkunetvfe_minkunethead_nusc_nosyncbn_sgd/train_on_trainset/best_nuScenes_lidarseg_mean_epoch_24.pth'  --seed 0 --eval 'segm'





