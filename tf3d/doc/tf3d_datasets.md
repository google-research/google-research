# 3D Scene Understanding Datasets

Working on scene understanding often involves working on different datasets with
various sensor configurations. We have come up with a solution to organize and
abstract these datasets, that allows us to readily use multiple 3D scene
understanding datasets with different sensor suits and different annotations.

For all our datasets we have the following recurring concept:

*   Frame: each entry contains frame level data like color and depth camera
    images, point cloud, camera intrinsics, groundtruth semantic and instance
    segmentations annotations.
*   Scene: each entry contains point-cloud/mesh data of a whole scene and a
    lightweight information to all frames in the scene.

Currently we are supporting three datasets:

*   [Waymo Open Dataset](waymo_open_dataset.md)
*   [Scannet Dataset](scannet_dataset.md)
*   [Rio Dataset](rio_dataset.md)

The data pipeline (`get_tf_data_dataset` in `tf3d/data_provider.py`) consists of
loading the `tfrecord` files, calling the task-specific `preprocessor` to
prepare the input and the labels, and various batching and parallel fetching
transformations.
