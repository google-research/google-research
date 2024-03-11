# Single Mesh Diffusion Models with Field Latents for Texture Generation

<img src="https://single-mesh-diffusion.github.io/images/teaser.png" width="800"/>

[Project Page](https://single-mesh-diffusion.github.io/) |
[Paper](https://arxiv.org/abs/2312.09250)

**Single Mesh Diffusion Models with Field Latents for Texture Generation**

Tommy Mitchel, Carlos Esteves, and Ameesh Makadia

Accepted to CVPR 2024.

## Usage instructions for sampling new textures for a labeled mesh.
Specify the paths to the<br>
(1) FLVAE checkpoint directory. *A pretrained model will be provided soon.*<br>
(2) FLDM checkpoint directory.<br>
(3) Base directory where individual preprocessed mesh subdirectories are
stored. *The format of preprocessed meshes and some downloadable examples will be
provided soon.*<br>
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
# "./" + <config.geom_record_path> + <config.obj_name>

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

### Citation

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
