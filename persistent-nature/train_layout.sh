# 1a. layout model
python train_layout.py \
	--training-mode layout \
	--cfg=stylegan2 --gpus=8 --batch 64 \
	--outdir training-runs/example-layout \
 	--data dataset/lhq_processed/img \
	--pose poses/width38.4_far16_noisy_height.pth \
	--img-resolution 32 --fov-mean 60 --fov-std 0 \
	--mirror=True --aug=noaug --batch-gpu 8 \
	--cmax 256 --cbase 32768 --map-depth 3 \
	--d-cbase 4096 --d-cmax 512 \
	--depth-scale 16 --depth-clip 20 --use-disp=True \
	--num-decoder-ch 32 --voxel-res 256 --voxel-size 0.15 --z-dim 128 \
	--feature-nerf=True --nerf-out-res 32 --nerf-out-ch 128 \
	--concat-depth=True --concat-acc=False --nerf-samples-per-ray 128 --nerf-far 16 \
	--gamma 0.01 --recon-weight 1000 --aug-policy translation,color,cutout \
	--metrics=fid5k_full --snap 50 

# 1b. layout model regularize geometry
# small finetuning step to help improve geometry, although FID will increase a bit; 
# it results in a cleaner sky mask for better compositing results with upsampler
python train_layout.py \
	--training-mode layout \
        --cfg=stylegan2 --gpus=8 --batch 64 \
	--outdir training-runs/example-layout-reg \
 	--data dataset/lhq_processed/img \
	--pose poses/width38.4_far16_noisy_height.pth \
	--img-resolution 32 --fov-mean 60 --fov-std 0 \
	--mirror=True --aug=noaug --batch-gpu 8 \
	--cmax 256 --cbase 32768 --map-depth 3 \
	--d-cbase 4096 --d-cmax 512 \
	--depth-scale 16 --depth-clip 20 --use-disp=True \
	--num-decoder-ch 32 --voxel-res 256 --voxel-size 0.15 --z-dim 128 \
	--feature-nerf=True --nerf-out-res 32 --nerf-out-ch 128 \
	--concat-depth=True --concat-acc=False --nerf-samples-per-ray 128 --nerf-far 16 \
	--gamma 0.01 --recon-weight 1000 --aug-policy translation,color,cutout \
	--metrics=fid5k_full --snap 50 \
        --concat-acc=True --use-wrapped-discriminator=True --kimg 400 \
        --ray-lambda-finite-difference 80.0 --ray-lambda-ramp-end 400 \
	--resume training-runs/example-layout/path-to-best-snapshot.pkl

# 2. upsampler model
# here; use layout model from a later snapshot with the refined sky mask; I used 400kimg in experiments
python train_layout.py \
	--training-mode upsampler \
        --cfg=stylegan2 --gpus=8 --batch 64 \
        --outdir training-runs/example-upsampler \
  	--data dataset/lhq_processed/img \
        --img-resolution 256 --fov-mean 60 --fov-std 0 \
        --mirror=True --aug=noaug --batch-gpu 8 \
        --cmax 128 --cbase 16384 --map-depth 2 --z-dim 128 \
        --metrics=fid5k_full --snap 50 --gamma 4.0 \
        --glr=0.0025 --dlr=0.0025  \
        --input-resolution 32 --D-ignore-depth-acc=True --concat-depth-and-acc=True \
	--lambda-rec 1.0 --lambda-up 5.0 \
        --lambda-gray-pixel 100 --lambda-gray-pixel-falloff 20 --use-3d-noise=True \
	--layout-model training-runs/example-layout-reg/path-to-experiment/network-snapshot-000400.pkl

# 3. sky model
python train_layout.py \
	--training-mode sky \
	--cfg=stylegan3-t --gpus=4 --batch 32 \
	--outdir training-runs/example-sky \
	--data dataset/lhq_processed/img \
	--img-resolution 256 \
	--mirror=True --aug=noaug --batch-gpu 8 \
	--metrics=fid5k_full --snap 50 --gamma=2 \
	--z-dim 128 --depth-clip 8 --mask-prob 0.5

