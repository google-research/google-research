# This package contains SMURF, a library for research on unsupervised learning of optical flow.

This code accompanies the paper **SMURF: Self-Teaching Multi-Frame Unsupervised RAFT with Full-Image Warping**. We hope that the code enables future research in unsupervised optical flow and beyond. If you find it useful in your own research, please give credit by citing our paper.

```
@inproceedings{stone2021smurf,
  title={SMURF: Self-Teaching Multi-Frame Unsupervised RAFT with Full-Image Warping},
  author={Stone, Austin and Maurer, Daniel and Ayvaci, Alper and Angelova, Anelia and Jonschkowski, Rico},
  booktitle={Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition
(CVPR)},
year={2021}
}
```

## Basic Usage

To train the network, run:

```
python3 -m smurf.smurf_main --train_on="<dataset-name>:<path-to-dataset>" --plot_dir=<path-to-plot-dir> --checkpoint_dir=<path-to-checkpoint>
```

To evaluate the network, run:

```
python3 -m smurf.smurf_main --eval_on="<dataset-name>:<path-to-evaluation-set>" --plot_dir=<path-to-plot-dir> --checkpoint_dir=<path-to-checkpoint>
```

Training and evaluation can also happen simultaneously by passing both the `--eval_on` and `--train_on` flags to smurf_main.

Currently, the datasets we support are KITTI ('kitti'), Sintel ('sintel'), Flying Chairs ('chairs'). You will need to download the datasets yourself and convert the data to tfrecord format using the scripts under the data_conversion_scripts folder.

For example, if you have the sintel dataset in tfrecord format located at /usr/local/home/datasets/sintel_clean, and you wanted to run with batch size 1 on images of size 384 X 512, you could run the following commands to both train and evaluate the network:

```
python3 -m smurf.smurf_main --train_on="sintel-clean:/usr/local/home/datasets/sintel_clean/test" --global_gpu_batch_size=1 --height=384 --width=512 --plot_dir='/usr/local/home/plots/smurf' --checkpoint_dir='/usr/local/home/checkpoints/smurf'
```

```
python3 -m smurf.smurf_main --eval_on="sintel-clean:/usr/local/home/datasets/sintel_clean/train" --global_gpu_batch_size=1 --height=384 --width=512 --plot_dir='/usr/local/home/plots/smurf' --checkpoint_dir='/usr/local/home/checkpoints/smurf'
```

Note that in the above case we train on the test set and evaluate on the train
set because the test set does not include training labels.

To export the flying chairs dataset to TF records, use:

```
python3 -m smurf.data_conversion_scripts.convert_flying_chairs_to_tfrecords --data_dir=<path to directory with chairs images and flow> --output_dir=<path to export directory>
```

You can optionally break the dataset into smaller files using the --shard and --num_shards parameters.


To export the Sintel dataset to TF records, use:

```
python3 -m smurf.data_conversion_scripts.convert_sintel_to_tfrecords --data_dir=<path to directory with chairs images and flow> --output_dir=<path to export directory>
```

To export the KITTI dataset to TF records, use:

```
python3 -m smurf.data_conversion_scripts.convert_KITTI_multiview_to_tfrecords --data_dir=<path to directory with chairs images and flow> --output_dir=<path to export directory>
```

Please note: To convert KITTI to TF records, use the scripts `convert_KITTI_flow_to_tfrecords.py` and `convert_KITTI_multiview_to_tfrecords.py` for the labeled flow images and the unlabeled multiview extension, respectively.


## Experimental Details

We used the following conditionals to decide the resolution, smoothness, and occlusion estimation during the experiments in our paper. If not specified below, we use the default settings.

```
if 'sintel' in train_dataset:
  height = 368
  width = 496
  weight_smooth1 = 2.5
  weight_smooth2 = 0.0
  # Scale all of the training steps to happen faster to avoid overfitting.
  lr_decay_after_num_steps: int(62500 * .2)
  lr_decay_steps: int(2500 * .2)
  num_train_steps: int(75000 * .2)
  occ_after_num_steps_brox: int(25000 * .2)
  selfsup_ramp_up_steps: int(6250 * .2)


if 'chairs' in train_dataset:
  height = 384
  width = 512
  weight_smooth1 = 2.0
  weight_smooth2 = 0.0

if 'kitti' in train_dataset:
  height = 296
  width = 696
  occlusion_estimation = 'brox'
```

After the model has been trained unsupervised using the default settings,
we enter multiframe training mode. We prepare labels using

```
python3 -m smurf.multiframe_training.main -- --input_dir=<path multiframe records> --output_dir=<path to output directory>
```

This will generate a new dataset with labels produced by a tiny per-frame
model which can then be used for "supervised" retraining. Note that the above
step will take an extremely long time on a single machine because it requires
retraining a tiny model per image frame. In practice we launch a distributed
job using many shards in parallel in order to complete this step in a reasonable
amount of time.

Use the following command to retrain on the multi-frame labels:

```
python3 -m smurf.smurf_main --train_on="multiframe-test-kitti:<path to records generate in prior step> --global_gpu_batch_size=1 --height=292 --width=696 --train_mode='sequence-supervised' --num_train_steps=30000 --lr_decay_after_num_steps=25000 --lr_decay_steps=1000
```

Note that although we train in mode `sequence-supervised`, we never use the
ground truth labels anywhere except for evaluation. The labels used in the prior
step come from the unsupervised model run on multiple frames as described in the
"Multi-Frame Self-Supervision" section of our paper.


Pre-trained model checkpoints
We provide checkpoints of trained models on the Sintel and KITTI dataset. The checkpoints are available on Google Cloud Storage:

* Chairs: [gs://gresearch/smurf/chairs-smurf](https://console.cloud.google.com/storage/browser/gresearch/smurf/chairs-smurf) (~60MB)

* Kitti: [gs://gresearch/smurf/kitti-smurf](https://console.cloud.google.com/storage/browser/gresearch/smurf/kitti-smurf) (~60MB)

* Sintel: [gs://gresearch/smurf/sintel-smurf](https://console.cloud.google.com/storage/browser/gresearch/smurf/sintel-smurf) (~60MB)

To use these checkpoints, download all files into a local checkpoint directory, e.g. /tmp/smurf/, either by using the Google Cloud Storage web interface or using gsutil:

```
gsutil cp -r gs://gresearch/smurf* /tmp/smurf/
```

You can then restore the checkpoints by passing
```
--checkpoint_dir=/tmp/smurf/kitti
```

There is a file called `apply_smurf.py` which can be used for simply running
smurf on a directory of images. See `apply_smurf.py` for more details.
