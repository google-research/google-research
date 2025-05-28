# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modified setup.py file for tiny-cuda-nn that allows building without GPU.

The default tiny-cuda-nn setup.py expects the build server to have a GPU
visible, in order to detect its compute capability.
"""

import glob
import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Find version of tinycudann by scraping CMakeLists.txt.
with open(os.path.join(ROOT_DIR, "CMakeLists.txt"), "r") as cmakelists:
  # pylint: disable=g-builtin-op
  for line in cmakelists.readlines():
    if line.strip().startswith("VERSION"):
      VERSION = line.split("VERSION")[-1].strip()
      break
  # pylint: enable=g-builtin-op

print(f"Building PyTorch extension for tiny-cuda-nn version {VERSION}")

ext_modules = []

include_networks = True
if "--no-networks" in sys.argv:
  include_networks = False
  sys.argv.remove("--no-networks")
  print("Building >> without << neural networks (just the input encodings)")

if os.name == "nt":

  def find_cl_path():
    for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
      glob_path = (r"C:\\Program Files (x86)\\Microsoft Visual "
                   r"Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64")
      paths = sorted(glob.glob(glob_path % edition), reverse=True)
      if paths:
        return paths[0]

  # If cl.exe is not on path, try to find it.
  if os.system("where cl.exe >nul 2>nul") != 0:
    cl_path = find_cl_path()
    if cl_path is None:
      raise RuntimeError(
          "Could not locate a supported Microsoft Visual C++ installation")
    os.environ["PATH"] += ";" + cl_path

# NOTE(jainajay): Update these device capability levels for different GPUs.
# Table of capabilities: https://developer.nvidia.com/cuda-gpus#compute.
# Ampere generation products support CC 8.0 and V100 supports 7.0.
major, minor = 8, 0
compute_capability = major * 10 + minor

nvcc_flags = [
    "-std=c++14",
    "--extended-lambda",
    "--expt-relaxed-constexpr",
    # The following definitions must be undefined
    # since TCNN requires half-precision operation.
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    # pylint: disable=line-too-long
    f"-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}",
    # pylint: enable=line-too-long
    f"-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}",
]
if os.name == "posix":
  cflags = ["-std=c++14"]
  nvcc_flags += [
      "-Xcompiler=-mf16c",
      "-Xcompiler=-Wno-float-conversion",
      "-Xcompiler=-fno-strict-aliasing",
  ]
elif os.name == "nt":
  cflags = ["/std:c++14"]

print(f"Targeting compute capability {compute_capability}")

definitions = [f"-DTCNN_MIN_GPU_ARCH={compute_capability}"]
nvcc_flags += definitions
cflags += definitions

# pylint: disable=line-too-long
# Some containers set this to contain old architectures that won't compile.
# We only need the one installed in the machine. See
# https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.CUDAExtension
# for docs.
# pylint: enable=line-too-long
os.environ["TORCH_CUDA_ARCH_LIST"] = " ".join(
    set([
        f"{major}.{minor}",
        f"{major}.{minor}+PTX",  # Forward compatibility.
    ]))

# List of sources.
bindings_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(bindings_dir, "../.."))
source_files = [
    "tinycudann/bindings.cpp",
    "../../src/cpp_api.cu",
    "../../src/common.cu",
    "../../src/common_device.cu",
    "../../src/encoding.cu",
]

if include_networks:
  source_files += [
      "../../src/network.cu",
      "../../src/cutlass_mlp.cu",
  ]

  if compute_capability >= 70:
    source_files.append("../../src/fully_fused_mlp.cu")
else:
  nvcc_flags.append("-DTCNN_NO_NETWORKS")
  cflags.append("-DTCNN_NO_NETWORKS")

ext = CUDAExtension(
    name="tinycudann_bindings._C",
    sources=source_files,
    include_dirs=[
        "%s/include" % root_dir,
        "%s/dependencies" % root_dir,
        "%s/dependencies/cutlass/include" % root_dir,
        "%s/dependencies/cutlass/tools/util/include" % root_dir,
    ],
    extra_compile_args={
        "cxx": cflags,
        "nvcc": nvcc_flags
    },
    libraries=["cuda", "cudadevrt", "cudart_static"],
)
ext_modules = [ext]

setup(
    name="tinycudann",
    version=VERSION,
    description="tiny-cuda-nn extension for PyTorch",
    long_description="tiny-cuda-nn extension for PyTorch",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA",
        "License :: BSD 3-Clause",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    keywords="PyTorch,cutlass,machine learning",
    url="https://github.com/nvlabs/tiny-cuda-nn",
    author="Thomas Müller, Jacob Munkberg, Jon Hasselgren, Or Perel",
    author_email="tmueller@nvidia.com, jmunkberg@nvidia.com, "
                 "jhasselgren@nvidia.com, operel@nvidia.com",
    maintainer="Thomas Müller",
    maintainer_email="tmueller@nvidia.com",
    download_url="https://github.com/nvlabs/tiny-cuda-nn",
    license="BSD 3-Clause \"New\" or \"Revised\" License",
    packages=["tinycudann"],
    install_requires=[],
    include_package_data=True,
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension})
