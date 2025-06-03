# my_python_utils, no patch
git clone https://github.com/mbaradad/my_python_utils

# SIREN
git clone https://github.com/dalmia/siren
rm -rf siren/.git
# copy files
rsync -av --exclude='*.patch' external/siren/ siren/
# apply patches
patch siren/siren/siren.py < external/siren/siren_py.patch

# PLENOXELS
git clone https://github.com/sxyu/svox2
mv svox2 plenoxels
rm -rf plenoxels/.git

# copy files
rsync -av --exclude='*.patch' external/plenoxels/ plenoxels/
# apply patches
patch plenoxels/environment.yml < external/plenoxels/environment.yml.patch
patch plenoxels/.gitignore < external/plenoxels/.gitignore.patch
patch plenoxels/opt/configs/syn.json < external/plenoxels/opt/configs/syn.json.patch
patch plenoxels/opt/opt.py < external/plenoxels/opt/opt.py.patch
patch plenoxels/opt/util/util.py < external/plenoxels/opt/util/util.py.patch
patch plenoxels/opt/util/dataset_base.py < external/plenoxels/opt/util/dataset_base.py.patch
patch plenoxels/opt/util/llff_dataset.py < external/plenoxels/opt/util/llff_dataset.py.patch
patch plenoxels/opt/util/nerf_dataset.py < external/plenoxels/opt/util/nerf_dataset.py.patch
patch plenoxels/opt/util/nsvf_dataset.py < external/plenoxels/opt/util/nsvf_dataset.py.patch
patch plenoxels/opt/util/config_util.py < external/plenoxels/opt/util/config_util.py.patch
patch plenoxels/svox2/svox2.py < external/plenoxels/svox2/svox2.py.patch
patch plenoxels/svox2/csrc/render_svox1_kernel.cu < external/plenoxels/svox2/csrc/render_svox1_kernel.cu.patch
patch plenoxels/svox2/utils.py < external/plenoxels/svox2/utils.py.patch


### PLENOCTREES
git clone https://github.com/sxyu/plenoctree
rm -rf plenoctree/.git

# copy files
rsync -av --exclude='*.patch' external/plenoctree/ plenoctree/
# apply patches
patch plenoctree/README.md < external/plenoctree/README.md.patch
patch plenoctree/nerf_sh/nerf/utils.py < external/plenoctree/nerf_sh/nerf/utils.py.patch
patch plenoctree/requirements.txt < external/plenoctree/requirements.txt.patch
patch plenoctree/environment.yml < external/plenoctree/environment.yml.patch
patch plenoctree/octree/nerf/utils.py < external/plenoctree/octree/nerf/utils.py.patch
patch plenoctree/octree/optimization.py < external/plenoctree/octree/optimization.py.patch
patch plenoctree/octree/extraction.py < external/plenoctree/octree/extraction.py.patch
patch plenoctree/octree/evaluation.py < external/plenoctree/octree/evaluation.py.patch

# SVOX 
git clone https://github.com/sxyu/svox
# move and rename files
rm -rf svox/.git
mv svox swvox
mv swvox/svox swvox/swvox
mv swvox/swvox/svox.py swvox/swvox/swvox.py
mv swvox/swvox/csrc/svox.cpp swvox/swvox/csrc/swvox.cpp
mv swvox/swvox/csrc/svox_kernel.cu swvox/swvox/csrc/swvox_kernel.cu
rm -rf swvox/docs swvox/.readthedocs.yaml
# copy files
rsync -av --exclude='*.patch' external/swvox/ swvox/
# apply patches
patch swvox/MANIFEST.in < external/swvox/MANIFEST.in.patch
patch swvox/README.md < external/swvox/README.md.patch
patch swvox/setup.py < external/swvox/setup.py.patch
patch swvox/swvox/csrc/CMakeLists.txt < external/swvox/swvox/csrc/CMakeLists.txt.patch
patch swvox/swvox/csrc/include/common.cuh < external/swvox/swvox/csrc/include/common.cuh.patch
patch swvox/swvox/csrc/include/data_spec.hpp < external/swvox/swvox/csrc/include/data_spec.hpp.patch
patch swvox/swvox/csrc/include/data_spec_packed.cuh < external/swvox/swvox/csrc/include/data_spec_packed.cuh.patch
patch swvox/swvox/csrc/rt_kernel.cu < external/swvox/swvox/csrc/rt_kernel.cu.patch
patch swvox/swvox/csrc/swvox.cpp < external/swvox/swvox/csrc/swvox.cpp.patch
patch swvox/swvox/csrc/swvox_kernel.cu < external/swvox/swvox/csrc/swvox_kernel.cu.patch
patch swvox/swvox/helpers.py < external/swvox/swvox/helpers.py.patch
patch swvox/swvox/__init__.py < external/swvox/swvox/__init__.py.patch
patch swvox/swvox/renderer.py < external/swvox/swvox/renderer.py.patch
patch swvox/swvox/swvox.py < external/swvox/swvox/swvox.py.patch

# Other small files:
wget https://raw.githubusercontent.com/ashawkey/torch-ngp/main/sdf/utils.py -O wavelets_sdf/sdf_utils.py
patch wavelets_sdf/sdf_utils.py < external/wavelets_sdf/sdf_utils.py.patch
wget https://raw.githubusercontent.com/tatp22/multidim-positional-encoding/master/positional_encodings/torch_encodings.py siren/torch_encodings.py

# to check patch correctly applied, use diff -bur with google dir:
# diff -bur siren/ external/google/siren/

# to create the patches:
# diff -u siren/siren/siren.py external/google/siren/siren/siren.py > external/siren/patches/siren_py.patch
