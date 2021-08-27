# TF Serving + ScaNN ops with Docker

**See [Docker Hub](https://hub.docker.com/r/google/tf-serving-scann) to use
these images.**

[TensorFlow Serving](https://github.com/tensorflow/serving) is a flexible and
efficient system for productionizing machine learning models. TF Serving makes
it easy to serve [SavedModels](https://www.tensorflow.org/guide/saved_model) for
inference. For more information, please see the TF Serving tutorials.

In order to deploy SavedModels with custom C++ TensorFlow ops, a custom build of
TF Serving that is linked to the custom ops is required (directions
[here](https://www.tensorflow.org/tfx/serving/custom_op)). ScaNN is an example
of such a custom op; here, we provide Docker images of custom TF Serving builds
linked to the ScaNN ops.

## Key differences from official TF Serving Docker images:

*   Linked to the ScaNN TF ops, which allows for deploying SavedModels that use
    ScaNN. **We highly recommend using the official TF Serving Docker images for
    SavedModels that don't use ScaNN.**
*   Compiled with the AVX2 and FMA3 instruction set extensions. These Docker
    images therefore **require CPUs that support these extensions (~2013 or
    later CPUs only).**
*   The official TF Serving Dockerfiles may be found
    [here](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/tools/docker)
    and their corresponding Docker images are on
    [Docker Hub](https://hub.docker.com/r/tensorflow/serving).

## Deployment

See instructions [here](https://www.tensorflow.org/tfx/serving/docker), but
replace any references to the `tensorflow/serving` Docker image with
`google/tf-serving-scann`. The list of all TF Serving + ScaNN images is listed
[here](https://hub.docker.com/r/google/tf-serving-scann). We currently do not
provide GPU or MKL versions of TF Serving.
