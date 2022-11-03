# MetaPose

**MetaPose: Fast 3D Pose from Multiple Views without 3D Supervision** </br>
*Conference on Computer Vision and Pattern Recognition (CVPR) 2022*</br>
<a href="https://arxiv.org/abs/2108.04869">arxiv</a> / <a href="https://metapose.github.io/">project page</a>

> In the era of deep learning, human pose estimation from multiple cameras with unknown calibration has received little attention to date. We show how to train a neural model to perform this task with high precision and minimal latency overhead. The proposed model takes into account joint location uncertainty due to occlusion from multiple views, and requires only 2D keypoint data for training. Our method outperforms both classical bundle adjustment and weakly-supervised monocular 3D baselines on the well-established Human3.6M dataset, as well as the more challenging in-the-wild Ski-Pose PTZ dataset.

<img src="https://metapose.github.io/task.png" alt="metapose task" style="width:600px;"/>

![MetaPose demo](https://storage.googleapis.com/gresearch/metapose/demo2.gif)

We provide more <a href="https://drive.google.com/drive/u/1/folders/1MrD9lvfz2keG58LoBx3X_pptBja-TQmY">[demo videos]</a> on Human36M, SkiPose and KTH Multiview Football II.

If you use any of this code or its derivatives, please consider citing our work:

```
@inproceedings{usman2021metapose,
    author    = {Usman, Ben and Tagliasacchi, Andrea and Saenko, Kate and Sud, Avneesh},
    title     = {MetaPose: Fast 3D Pose from Multiple Views without 3D Supervision},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022}
}
```

Contact: usmn@google.com

### Checking out the code

```
svn export https://github.com/google-research/google-research/trunk/metapose
```

### Data + checkpoints

```
wget http://storage.googleapis.com/gresearch/metapose/metapose.tar
tar xvf metapose.tar
export DATA_ROOT=$(pwd)/metapose
```

**Disclaimer**: the test split of Human3.6M (subjects 9+11) used by the majority of prior work to compare the performance of pose estimation models contains only male actors; in the future, this might promote work that fails silently on female subjects; we would like to encourage authors to explore more balanced testing setups in future research.

### Requirements

All dependencies can be installed via `pip install -r requirements.txt`. 
The code was tested to work with

```
tensorflow==2.8.0
tensorflow-datasets==4.5.2
tensorflow-probability==0.16.0
aniposelib==0.4.3
opencv-contrib-python==4.6.0.66
```
### Train

```
N_CAM=4  # or 3, or 2

python -m metapose.train_metapose \
    --data_root=${DATA_ROOT} \
    --experiment_name=train_h36m_cam1_${N_CAM} \
    --tb_log_dir=/tmp/train_h36m_cam1_${N_CAM} \
    --dataset=data/h36m/opt/${N_CAM} \
    --data_splits=train,test \
    --n_cam=${N_CAM} \
    --train_repeat_k=1 \
    --permute_cams_aug=false \
    --use_equivariant_model=true \
    --standardize_init_best=false \
    --main_mlp_spec=512,512,ccat,512,512,ccat,512 \
    --early_stopping_patience=500 \
    --epochs_per_stage=3000
```

Due to licensing restrictions, we cannot release data or checkpoints to
reproduce ski-pose experiments, however we provide the hyperparameters that we
observed to give the best performance.

```
N_CAM=6  # or 4, or 2

python -m metapose.train_metapose \
    --data_root=${DATA_ROOT} \
    --experiment_name=train_skipose_cam1_${N_CAM} \
    --tb_log_dir=/tmp/train_h36m_cam1_${N_CAM} \
    --dataset=data/ski/opt/${N_CAM} \
    --data_splits=train,test \
    --n_cam=${N_CAM} \
    --train_repeat_k=1 \
    --permute_cams_aug=false \
    --use_equivariant_model=true \
    --standardize_init_best=true \
    --main_mlp_spec=512,512,ccat,512,512,ccat,512,512,ccat,512,512,ccat,512
```

### Evaluate using a pre-trained checkpoint and generate predictions

```
N_CAM=4  # or 3, or 2
SAVE_PRED_PATH=/tmp/preds

python -m metapose.train_metapose \
    --data_root=${DATA_ROOT} \
    --experiment_name=test_github_version \
    --tb_log_dir=/tmp/ \
    --dataset=data/h36m/opt/${N_CAM} \
    --data_splits=train,test \
    --dataset_warmup=false \
    --n_cam=${N_CAM} \
    --train_repeat_k=1 \
    --permute_cams_aug=false \
    --use_equivariant_model=true \
    --debug_show_single_frame_pmpjes=false \
    --debug_enable_check_numerics=false \
    --use_equivariant_model=true \
    --standardize_init_best=false \
    --main_mlp_spec=512,512,ccat,512,512,ccat,512 \
    --load_weights_from=${DATA_ROOT}/ckpt/h36m/cam${N_CAM}/model \
    --load_stages_n=1 \
    --epochs_per_stage=0 \
    --max_stage_attempts=1 \
    --save_preds_to=${SAVE_PRED_PATH}
```


### Run iterative refinement (S1+IR)

```
python -m metapose.launch_iterative_solver \
    --input_path=${DATA_ROOT}/data/h36m/pre/train \
    --output_path=/tmp/test_infer_opt
```
(use `--fake_gt_init`, `--use_weak_repr`, or `--gt_heatmaps` to emulate perfect
initialization, matching camera model, or perfect heatmaps)

### Run iterative refinement with GT init (GT+IR)
```
python -m metapose.launch_iterative_solver \
    --input_path=${DATA_ROOT}/data/h36m/pre/train \
    --output_path=/tmp/test_infer_opt /
    --fake_gt_init=true
```

### Run AniPose
```
python -m metapose.launch_anipose
    --input_path=${DATA_ROOT}/data/h36m/pre/train \
    --output_path=/tmp/test_anipose
```

For both `launch_iterative_solver` and `launch_anipose` use `--cam_subset=0,1`
flag to restrict inference to a subset of cameras.

## Ablations

### Self-supervised training (S1+S2/SS)
```
N_CAM=4  # or 3, or 2

python -m metapose.train_metapose \
    --data_root=${DATA_ROOT} \
    --experiment_name=train_h36m_cam1_${N_CAM} \
    --tb_log_dir=/tmp/train_h36m_cam1_${N_CAM} \
    --dataset=data/h36m/opt/${N_CAM} \
    --data_splits=train,test \
    --n_cam=${N_CAM} \
    --train_repeat_k=1 \
    --permute_cams_aug=false \
    --use_equivariant_model=true \
    --standardize_init_best=false \
    --main_mlp_spec=512,512,ccat,512,512,ccat,512 \
    --early_stopping_patience=500 \
    --epochs_per_stage=3000 \
    --lambda_fwd_loss=0.0 \
    --lambda_logp_loss=1.0 \
    --lambda_xopt_loss=0.0
```
(default values are `lambda_fwd_loss=1.0, lambda_logp_loss=0.0,
lambda_xopt_loss=0.0`)

### Student-teacher (S1+S2/TS)

```
N_CAM=4  # or 3, or 2

python -m metapose.train_metapose \
    --data_root=${DATA_ROOT} \
    --experiment_name=train_h36m_cam1_${N_CAM} \
    --tb_log_dir=/tmp/train_h36m_cam1_${N_CAM} \
    --dataset=data/h36m/opt/${N_CAM} \
    --data_splits=train,test \
    --n_cam=${N_CAM} \
    --train_repeat_k=1 \
    --permute_cams_aug=false \
    --use_equivariant_model=true \
    --standardize_init_best=false \
    --main_mlp_spec=512,512,ccat,512,512,ccat,512 \
    --early_stopping_patience=500 \
    --epochs_per_stage=3000 \
    --lambda_fwd_loss=0.0 \
    --lambda_logp_loss=0.0 \
    --lambda_xopt_loss=1.0
```
(default values are `lambda_fwd_loss=1.0, lambda_logp_loss=0.0,
lambda_xopt_loss=0.0`)

### Drop some inputs to the equivariant MLP
```
N_CAM=4  # or 3, or 2

python -m metapose.train_metapose \
    --data_root=${DATA_ROOT} \
    --experiment_name=train_h36m_cam1_${N_CAM} \
    --tb_log_dir=/tmp/train_h36m_cam1_${N_CAM} \
    --dataset=data/h36m/opt/${N_CAM} \
    --data_splits=train,test \
    --n_cam=${N_CAM} \
    --train_repeat_k=1 \
    --permute_cams_aug=false \
    --use_equivariant_model=true \
    --standardize_init_best=false \
    --main_mlp_spec=512,512,ccat,512,512,ccat,512 \
    --early_stopping_patience=500 \
    --epochs_per_stage=3000 \
    --debug_drop_mlp_inputs=pose,cams,heatmaps
```
(default value is `--debug_drop_mlp_inputs=`)

### Use bone lengths (S1+S2/BL)
```
N_CAM=4  # or 3, or 2

python -m metapose.train_metapose \
    --data_root=${DATA_ROOT} \
    --experiment_name=train_h36m_cam1_${N_CAM} \
    --tb_log_dir=/tmp/train_h36m_cam1_${N_CAM} \
    --dataset=data/h36m/opt/${N_CAM} \
    --data_splits=train,test \
    --n_cam=${N_CAM} \
    --train_repeat_k=1 \
    --permute_cams_aug=false \
    --use_equivariant_model=true \
    --standardize_init_best=false \
    --main_mlp_spec=512,512,ccat,512,512,ccat,512 \
    --early_stopping_patience=500 \
    --epochs_per_stage=3000 \
    --use_bone_len=true \
    --lambda_limb_len_loss=1.0
```
(default values are `--use_bone_len=false --lambda_limb_len_loss=0.0`)


### Use non-equivariant MLP (S1+S2/MLP)
```
N_CAM=4  # or 3, or 2

python -m metapose.train_metapose \
    --data_root=${DATA_ROOT} \
    --experiment_name=train_h36m_cam1_${N_CAM} \
    --tb_log_dir=/tmp/train_h36m_cam1_${N_CAM} \
    --dataset=data/h36m/opt/${N_CAM} \
    --data_splits=train,test \
    --n_cam=${N_CAM} \
    --train_repeat_k=1 \
    --permute_cams_aug=false \
    --use_equivariant_model=false \
    --model_mlp_sizes=512,512,512,128 \
    --standardize_init_best=false \
    --early_stopping_patience=500 \
    --epochs_per_stage=3000
```
(default values are `--use_equivariant_model=true`, also uses
`--model_mlp_sizes` instead of `--main_mlp_spec`)