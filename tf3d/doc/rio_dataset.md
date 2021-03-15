# [Rio] dataset

We use <a href='https://www.tensorflow.org/datasets' target='_blank'>tensorflow_datasets</a> (tfds) API to
specify our underlying tf.train.Example proto sstables/tfrecords.

We export raw [Rio] data to [Example] protos of two different kinds:

*   `frame`: Each [Example] proto contains frame level data like color and depth
    camera image, camera intrinsics, groundtruth semantic and instance
    segmentations annotations.
*   `scene`: Here each [Example] proto contains pointcloud/mesh data of a whole
    scene and a lightweight information to all frames in the scene.

Please refer to the <a href='https://github.com/google-research/google-research/blob/master/tf3d/datasets/specs/rio_specs.py' target='_blank'>rio_specs.py</a> for more details.

## Download

Please follow the instructions at the <a href='https://waldjohannau.github.io/RIO/' target='_blank'>Rio Dataset Website</a> for registration and
downloading of the tfrecords.

[Rio]: https://waldjohannau.github.io/RIO/
[Example]: https://www.tensorflow.org/api_docs/python/tf/train/Example
