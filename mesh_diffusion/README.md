# Single Mesh Diffusion Models with Field Latents for Texture Generation

[Project Page](https://single-mesh-diffusion.github.io/) |
[Paper](https://arxiv.org/abs/2312.09250)

**Single Mesh Diffusion Models with Field Latents for Texture Generation**

Tommy Mitchel, Carlos Esteves, and Ameesh Makadia

Accepted to CVPR 2024.

## Citation

If you found this work useful, please cite

```
@article{mitchel2023meshdiffusion,
  author        = {Mitchel, Thomas and Esteves, Carlos and Makadia, Ameesh},
  title         = {Single Mesh Diffusion Models with Field Latents for Texture Generation},
  year          = {2023},
  eprint        = {2312.09250},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```
## Usage instructions for sampling new textures for a labeled mesh.
Specify the paths to the<br>
(1) FLVAE checkpoint directory. *A pretrained model will be provided soon.*<br>
(2) FLDM checkpoint directory.<br>
(3) Base directory where individual preprocessed mesh subdirectories are
stored. *The instructions and code for preprocessing meshes is [available
here](https://github.com/twmitchel/field_latent_preprocess/tree/main).*<br>
(4) Mesh (subdirectory) name.<br>
(5) Path to ply file that contains the labels.<br>
(6) Path to output directory where generated textures should be saved.

All the paths in this example should be relative to the directory above
`mesh_diffusion`.

```python 
# FLVAE checkpoint directory.
config.trainer_checkpoint_dir = 'mesh_diffusion/path/to/flvae_checkpoint/dir/'

# FLDM checkpoint directory.
config.model_checkpoint_dir = 'mesh_diffusion/path/to/fldm_checkpoint/dir/'

# Preprocessed mesh files should be under
# ./<config.geom_record_path>/<config.obj_name>

# Base directory where individual preprocess mesh subdirectories are stored.
config.geom_record_path='mesh_diffusion/path/to/meshes/'

# Mesh (subdirectory) name
config.obj_name='mesh_name'

# Path to ply file that contains the labels.
config.obj_labels='mesh_diffusion/path/to/mesh_labeled.ply'

# Path to output directory where generated textures should be saved.
config.sample_save_dir = 'mesh_diffusion/generated_texture_samples/'
```

To generate new sampels, run the following command line from the directory above
`mesh_diffusion`. Note the sampling config file is specified through the
`--config` command line flag.

```
>> python -m mesh_diffusion.sin_mesh_ddm.main --config=mesh_diffusion/sin_mesh_ddm/configs/default_sample_labeled.py --workdir=mesh_diffusion/workdir/ --logtostderr
```

# Data and Models

## Data and models for a single mesh example run
For demonstration purposes, we have included the preprocessed data and a trained FLDM model on the *labeled sea lion mesh* shown in the paper's experiments.

#### Data source
The original sea lion mesh was selected from the Objaverse dataset, created by user *@edemaistre* with CC-Attribution license ([Objaverse link](https://objaverse.allenai.org/explore/?query=fe973fc8e8c049c6b9ab884137fc9463), [Sketchfab link to model and license info](https://sketchfab.com/3d-models/another-sea-lion-statue-in-san-francisco-2-fe973fc8e8c049c6b9ab884137fc9463)).

#### Preprocessed mesh data
- As described in the paper (see the Experiments section), for training we create 500 copies of a given mesh each with ~30K vertices and different triangulations to prevent the denoising network from learning the connectivity.  For the sea lion example, we have provided these remeshings hosted in a google cloud storage bucket (**[link](https://storage.googleapis.com/geometric-ai-public/mesh_diffusion/preprocessed_meshes/fe973fc8e8c049c6b9ab884137fc9463/fe973fc8e8c049c6b9ab884137fc9463_preprocessed.zip)**). Note, this file is very large (65GB) since in addition to the meshes it holds all the precomputed connectivity information used by the FLDM architecture.

#### Labeled mesh
- **[PLY file](https://storage.googleapis.com/geometric-ai-public/mesh_diffusion/preprocessed_meshes/fe973fc8e8c049c6b9ab884137fc9463/fe973fc8e8c049c6b9ab884137fc9463_sel.ply)** with *labels*. This file modifies the original sea lion mesh described above ([link](https://sketchfab.com/3d-models/another-sea-lion-statue-in-san-francisco-2-fe973fc8e8c049c6b9ab884137fc9463), [CC Attribution license](https://creativecommons.org/licenses/by/4.0/)) to include labels (see Figure 4 in the [paper](https://arxiv.org/abs/2312.09250)).
- This file path should be specified in the `config.obj_labels` flag for training or sampling.

### Pre-trained FL-VAE
The FL-VAE was trained following the details provided in the paper (Experiments -- Pre-training the FL-VAE).

- [FL-VAE checkpoint](https://storage.googleapis.com/geometric-ai-public/mesh_diffusion/models/flvae/flvae_checkpoint.zip)

### Pre-trained FLDM model

- [FLDM checkpoint](https://storage.googleapis.com/geometric-ai-public/mesh_diffusion/models/fldm/fldm_checkpoint_fe973fc8e8c049c6b9ab884137fc9463.zip) generated after 200K training steps (running FLDM training with `--config=default_train_labeled.py`).  This checkpoint can be used to run the sampling script (specify the location in `config.model_checkpoint_dir`).
- The contents of this zip file should be placed under `<config.geom_record_path>/<config.obj_name>`.
