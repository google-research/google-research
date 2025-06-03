# 1a. train triplane model
python train_triplane.py --outdir=./training-runs/example-triplane \
        --cfg=lhq --data=./dataset/lhq_processed_for_triplane_cam050 \
        --gpus=8 --batch=32 --gamma=10 --batch-gpu=4 \
	--cmax=256 --gen_pose_cond=False --disc_c_noise=1 \
        --white_sky=True --ignore_LR_disp=False \
        --depth_clip=20 --depth_scale=16  --ignore_w_to_upsampler=True

# 1b. small finetuning step to remove opaque regions with sky color
python train_triplane.py --outdir=./training-runs/example-triplane-regularized \
        --cfg=lhq --data=./dataset/lhq_processed_for_triplane_cam050 \
        --gpus=8 --batch=32 --gamma=10 --batch-gpu=4 \
	--cmax=256 --gen_pose_cond=False --disc_c_noise=1 \
        --white_sky=True --ignore_LR_disp=False \
        --depth_clip=20 --depth_scale=16  --ignore_w_to_upsampler=True \
        --lambda_sky_pixel=40 --lambda_ramp_end=400 --kimg=400 \
	--resume=training-runs/example-triplane/path-to-best-snapshot.pkl

# for sky model, see step 3. in train_layout.sh (uses the same sky model)
