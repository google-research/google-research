# [ScanNet] dataset

We use <a href='https://www.tensorflow.org/datasets' target='_blank'>tensorflow_datasets</a> (tfds) API to
specify our underlying tf.train.Example proto sstables/tfrecords.

We export raw [ScanNet] data to [Example] protos of two different kinds:

*   `frame`: Each [Example] proto contains frame level data like color and dpeth
    camera image, camera intrinsics, groundtruth semantic and instance
    segmentations annotations.
*   `scene`: Here each [Example] proto contains pointcloud/mesh data of a whole
    scene and a lightweight information to all frames in the scene.

Please refer to the
[scannet_specs.py](https://github.com/google-research/google-research/blob/master/tf3d/datasets/specs/scannet_specs.py)
for more details.

## Download

Please follow the instructions at the
[ScanNet Dataset Website](http://www.scan-net.org/) for registration and
downloading of the tfrecords. Once you have registered, there is a 
`--tf_semantic` flag that you can use with the ScanNet download script to 
download the tf records.

[ScanNet]: http://www.scan-net.org/
[Example]: https://www.tensorflow.org/api_docs/python/tf/train/Example
