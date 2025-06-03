### stylegan libraries
git clone https://github.com/NVlabs/stylegan3.git libraries/stylegan3
ln -s libraries/stylegan3/dnnlib ./ 
ln -s libraries/stylegan3/torch_utils ./ 
ln -s libraries/stylegan3/legacy.py ./ 

# anyres GAN --> sky
wget https://raw.githubusercontent.com/chail/anyres-gan/bccdf6069742d449c2a42fdfbdd0ceb2270544c2/training/networks_stylegan3.py -O external/stylegan/training/networks_stylegan3_sky.py
patch external/stylegan/training/networks_stylegan3_sky.py external/stylegan/training/networks_stylegan3_sky.patch

# sgan3 --> networks_sgan2
wget https://raw.githubusercontent.com/NVlabs/stylegan3/583f2bdd139e014716fc279f23d362959bcc0f39/training/networks_stylegan2.py -O external/stylegan/training/networks_stylegan2_terrain.py
patch external/stylegan/training/networks_stylegan2_terrain.py external/stylegan/training/networks_stylegan2_terrain.patch

### gsn model utils, nerf utils, layers, generator, discriminator
wget https://raw.githubusercontent.com/apple/ml-gsn/07044f0b7f3649c2e66a3bf4d2c7f0c75a42f399/models/model_utils.py -P external/gsn/models/
patch external/gsn/models/model_utils.py external/gsn/models/model_utils.patch
wget https://raw.githubusercontent.com/apple/ml-gsn/07044f0b7f3649c2e66a3bf4d2c7f0c75a42f399/models/nerf_utils.py -P external/gsn/models/
patch external/gsn/models/nerf_utils.py external/gsn/models/nerf_utils.patch
wget https://raw.githubusercontent.com/apple/ml-gsn/07044f0b7f3649c2e66a3bf4d2c7f0c75a42f399/models/layers.py -P external/gsn/models/
patch external/gsn/models/layers.py external/gsn/models/layers.patch
wget https://raw.githubusercontent.com/apple/ml-gsn/07044f0b7f3649c2e66a3bf4d2c7f0c75a42f399/models/generator.py -P external/gsn/models/
patch external/gsn/models/generator.py external/gsn/models/generator.patch
wget https://raw.githubusercontent.com/apple/ml-gsn/07044f0b7f3649c2e66a3bf4d2c7f0c75a42f399/models/discriminator.py -P external/gsn/models/
patch external/gsn/models/discriminator.py external/gsn/models/discriminator.patch

### triplane model and rendering utils
# rendering utils
wget https://raw.githubusercontent.com/NVlabs/eg3d/60da7af6eb46ba8484d69d4d7644c4b640a97084/eg3d/training/volumetric_rendering/renderer.py -P external/eg3d/training/volumetric_rendering/
patch external/eg3d/training/volumetric_rendering/renderer.py  external/eg3d/training/volumetric_rendering/renderer.patch
wget https://raw.githubusercontent.com/NVlabs/eg3d/60da7af6eb46ba8484d69d4d7644c4b640a97084/eg3d/training/volumetric_rendering/ray_marcher.py -P external/eg3d/training/volumetric_rendering/
patch external/eg3d/training/volumetric_rendering/ray_marcher.py  external/eg3d/training/volumetric_rendering/ray_marcher.patch
wget https://raw.githubusercontent.com/NVlabs/eg3d/60da7af6eb46ba8484d69d4d7644c4b640a97084/eg3d/training/volumetric_rendering/ray_sampler.py -P external/eg3d/training/volumetric_rendering/
patch external/eg3d/training/volumetric_rendering/ray_sampler.py  external/eg3d/training/volumetric_rendering/ray_sampler.patch
# rendering utils, no changes
wget https://raw.githubusercontent.com/NVlabs/eg3d/60da7af6eb46ba8484d69d4d7644c4b640a97084/eg3d/training/volumetric_rendering/math_utils.py -P external/eg3d/training/volumetric_rendering/
wget https://raw.githubusercontent.com/NVlabs/eg3d/60da7af6eb46ba8484d69d4d7644c4b640a97084/eg3d/training/volumetric_rendering/__init__.py -P external/eg3d/training/volumetric_rendering/
# models
wget https://raw.githubusercontent.com/NVlabs/eg3d/60da7af6eb46ba8484d69d4d7644c4b640a97084/eg3d/training/triplane.py -P external/eg3d/training/
patch external/eg3d/training/triplane.py  external/eg3d/training/triplane.patch
wget https://raw.githubusercontent.com/NVlabs/eg3d/60da7af6eb46ba8484d69d4d7644c4b640a97084/eg3d/training/superresolution.py -P external/eg3d/training/
patch external/eg3d/training/superresolution.py  external/eg3d/training/superresolution.patch

### training - layout model stylegan pipeline
# dataset.py, loss.py, augment.py (no changes), training_loop.py
wget https://raw.githubusercontent.com/NVlabs/stylegan3/583f2bdd139e014716fc279f23d362959bcc0f39/training/dataset.py -P external/stylegan/training/
patch external/stylegan/training/dataset.py external/stylegan/training/dataset.patch
wget https://raw.githubusercontent.com/NVlabs/stylegan3/583f2bdd139e014716fc279f23d362959bcc0f39/training/loss.py -P external/stylegan/training/
patch external/stylegan/training/loss.py external/stylegan/training/loss.patch
wget https://raw.githubusercontent.com/NVlabs/stylegan3/583f2bdd139e014716fc279f23d362959bcc0f39/training/augment.py -P external/stylegan/training/
wget https://raw.githubusercontent.com/NVlabs/stylegan3/583f2bdd139e014716fc279f23d362959bcc0f39/training/training_loop.py -P external/stylegan/training/
patch external/stylegan/training/training_loop.py external/stylegan/training/training_loop.patch
# copy metrics folder from downloaded stylegan library
cp -r libraries/stylegan3/metrics external/stylegan/
# metric_main.py, metric_utils.py
patch external/stylegan/metrics/metric_main.py external/stylegan/metrics/metric_main.patch
patch external/stylegan/metrics/metric_utils.py external/stylegan/metrics/metric_utils.patch
# train.py
wget https://raw.githubusercontent.com/NVlabs/stylegan3/583f2bdd139e014716fc279f23d362959bcc0f39/train.py -O train_layout.py
patch train_layout.py train_layout.patch
# diff augment library from GSN (unchanged)
wget https://raw.githubusercontent.com/apple/ml-gsn/07044f0b7f3649c2e66a3bf4d2c7f0c75a42f399/models/diff_augment.py -P external/gsn/models/

### training - triplane model eg3d pipeline
# dataset.py, loss.py, augment.py (no changes), training_loop.py, dual_discriminator.py
wget https://raw.githubusercontent.com/NVlabs/eg3d/60da7af6eb46ba8484d69d4d7644c4b640a97084/eg3d/training/dataset.py -P external/eg3d/training/
patch external/eg3d/training/dataset.py  external/eg3d/training/dataset.patch
wget https://raw.githubusercontent.com/NVlabs/eg3d/60da7af6eb46ba8484d69d4d7644c4b640a97084/eg3d/training/loss.py -P external/eg3d/training/
patch external/eg3d/training/loss.py  external/eg3d/training/loss.patch
wget https://raw.githubusercontent.com/NVlabs/eg3d/60da7af6eb46ba8484d69d4d7644c4b640a97084/eg3d/training/augment.py -P external/eg3d/training/
wget https://raw.githubusercontent.com/NVlabs/eg3d/60da7af6eb46ba8484d69d4d7644c4b640a97084/eg3d/training/training_loop.py -P external/eg3d/training/
patch external/eg3d/training/training_loop.py  external/eg3d/training/training_loop.patch
wget https://raw.githubusercontent.com/NVlabs/eg3d/60da7af6eb46ba8484d69d4d7644c4b640a97084/eg3d/training/dual_discriminator.py -P external/eg3d/training/
patch external/eg3d/training/dual_discriminator.py  external/eg3d/training/dual_discriminator.patch
# train.py
wget https://raw.githubusercontent.com/NVlabs/eg3d/60da7af6eb46ba8484d69d4d7644c4b640a97084/eg3d/train.py -O train_triplane.py
patch train_triplane.py train_triplane.patch
