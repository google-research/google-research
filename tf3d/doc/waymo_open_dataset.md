# [Waymo Open dataset]

We use [tensorflow_datasets](https://www.tensorflow.org/datasets) (tfds) API to
specify our underlying tf.train.Example proto sstables/tfrecords.

We use `scene` to describe each 20 second snippet of car's journey, whereas
`frame` is used to describe data collected within a small time range
(approximately single timestamp).

All per `frame` data (e.g lidar points, camera extrinsics, object pose) are
provided in a local vehicle reference frame. Each vehicle frame poses are
provided w.r.t a global world frame for a particular `scene`.

Please refer to the
[specs/waymo_frames.py](https://github.com/google-research/google-research/blob/master/tf3d/datasets/specs/waymo_frames.py)
and the
[specs/waymo_scenes.py](https://github.com/google-research/google-research/blob/master/tf3d/datasets/specs/waymo_scenes.py)
for a more detailed specification of the dataset structure.

## Download

Here are the instructions for downloading the Waymo Open Dataset tfrecords into
a local directory.

### Prerequisite

*   Register with your gmail account to use the
    [Waymo Open Dataset](https://waymo.com/open/). Please <b>Note</b> that you 
    will not be able to download the tfreocrds without having registered to the 
    Waymo Open Dataset.
*   Python 2 or 3
*   gsutil (command line tool to download files from google cloud. Follow this
    [Guide](https://cloud.google.com/sdk/docs) to install. After installing the
    Cloud SDK, simply run `gcloud init`, then you will be asked to log into your
    google account. Make sure it is the same account that has Waymo Dataset
    access.)
*   Make sure you have at least 2TB of storage on the disk to which you are
    downloading the data.

### Binary for Downloading the tfrecords

Run
[waymo_frames_batch_download.py](https://github.com/google-research/google-research/blob/master/tf3d/datasets/tools/waymo_frames_batch_download.py)
binary to download the tfrecords into your local directory.

```
python waymo_frames_batch_download.py --target_dir /tmp/dataset/waymo
```

[Waymo Open dataset]: https://waymo.com/open/
