1. Download the [Tensorflow repo](https://github.com/tensorflow/tensorflow) to a local folder `tensorflow`.
1. Create a folder at `tf3d/ops/third_party`, and copy the following files / folder from `tensorflow/third_party` folder:
   * `eigen3`
   * `toolchains`
   * `BUILD`
   * `com_google_absl_fix_mac_and_nvcc_build.patch`
   * `com_google_absl.BUILD`
   * `cub.BUILD`
   * `eigen.BUILD`
   * `repo.bzl`
1. Download the [TensorFlow Custom Op repo](https://github.com/tensorflow/custom-op) to a local folder `tf_custom_op`.
1. Copy the files / folders to `tf3d/ops` folder from `tf_custom_op` folder:
   * `gpu`
   * `tf`
   * `configure.sh`
1. Following the repo's guidance, set up the nvidia-docker.
1. Within the docker image, enter `tf3d/ops` folder and run the following to test the building:

   `./configure.sh`

   `bazel run sparse_conv_ops_py_test  --experimental_repo_remote_exec`


