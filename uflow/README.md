# This package contains UFlow, a library for research on learning unsupervised optical flow.

Read paper at **insert link here.**


## Basic Usage

To train the network, run:

`
python3 -m uflow.uflow_main --train_on="<dataset-name>:<path-to-dataset>" --plot_dir=<path-to-plot-dir> --checkpoint_dir=<path-to-checkpoint>
`

To evaluate the network, run:
`
python3 -m uflow.uflow_evaluator --eval_on="<dataset-name>:<path-to-evaluation-set>" --plot_dir=<path-to-plot-dir> --checkpoint_dir=<path-to-checkpoint>
`

Currently, the datasets we support are KITTI ('kitti'), Sintel ('sintel'), Flying Chairs ('chairs'). You will need to download the datasets yourself and convert the data to tfrecord format using the scripts under the data_conversion_scripts folder. We also support converting your own video to a dataset using the script misc/convert_video_to_dataset.py.

For example, if you have the sintel dataset in tfrecord format located at /usr/local/home/datasets/sintel_clean, and you wanted to run with batch size 1 on images of size 384 X 512, you could run the following commands to both train and evaluate the network:

`
python3 -m uflow.uflow_main --train_on="sintel-clean:/usr/local/home/datasets/sintel_clean/test" --batch_size=1 --height=384 --width=512 --plot_dir='/usr/local/home/plots/uflow' --checkpoint_dir='/usr/local/home/checkpoints/uflow'
`

`
python3 -m uflow.uflow_evaluator --eval_on="sintel-clean:/usr/local/home/datasets/sintel_clean/train" --batch_size=1 --height=384 --width=512 --plot_dir='/usr/local/home/plots/uflow' --checkpoint_dir='/usr/local/home/checkpoints/uflow'
`

The uflow_evaluator should be run simultaneously with uflow_main in order to train and evaluate at the same time. Note that in the above case we train on the test set and evaluate on the train
set because the test set does not include training labels.

To convert your own video to a trainable dataset, run (for example):

`
python3 -m uflow.misc.convert_video_to_dataset --video_path=files/billiards_clip.mp4 --output_path=/tmp/billiards_dataset
`

You can then train UFlow on this dataset using:
`
python3 -m uflow.uflow_main --train_on="custom:/tmp/billiards_dataset" --batch_size=1 --height=384 --width=512 --plot_dir='/usr/local/home/plots/uflow' --checkpoint_dir='/usr/local/home/checkpoints/uflow'
`

To export the flying chairs dataset to TF records, use:
`
python3 -m uflow.data_conversion_scripts.convert_flying_chairs_to_tfrecords --data_dir=<path to directory with chairs images and flow> --output_dir=<path to export directory> --shard=0 --num_shards=1
`
You can optionally break the dataset into smaller files using the --shard and --num_shards parameters.


To export the Sintel dataset to TF records, use:
`
python3 -m uflow.data_conversion_scripts.convert_sintel_to_tfrecords --data_dir=<path to directory with chairs images and flow> --output_dir=<path to export directory> --shard=0 --num_shards=1
`

To export the KITTI dataset to TF records, use:
`
python3 -m uflow.data_conversion_scripts.convert_KITTI_multiview_to_tfrecords --data_dir=<path to directory with chairs images and flow> --output_dir=<path to export directory> --shard=0 --num_shards=1
`

Please note: To convert KITTI to TF records, use the scripts convert_KITTI_flow_to_tfrecords.py and convert_KITTI_multiview_to_tfrecords.py for the labeled flow images and the unlabeled multiview extension, respectively.


## Experimental Details

We used the following conditionals to decide the resolution,
augmentation, smoothness, and occlusion estimation during the experiments in our paper.

`
if 'sintel' in train_dataset:
  augment_flip_up_down = True
  height = 448
  width = 1024
  weight_smooth1 = 4.0
  weight_smooth2 = 0.0
  occlusion_estimation = 'wang'

if 'chairs' in train_dataset:
  height = 384
  width = 512
  weight_smooth1 = 4.0
  weight_smooth2 = 0.0
  occlusion_estimation = 'wang'

if 'kitti' in train_dataset:
  height = 640
  width = 640
  weight_smooth1 = 0.0
  weight_smooth2 = 2.0
  occlusion_estimation = 'brox'
