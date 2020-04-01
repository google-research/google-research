# Open source release of [EMCAPSNet](https://openreview.net/pdf?id=HJWLfGWRb)

This directory contains a preliminary release of the EMCAPSNet model and is under construction.

Training a model on smallNORB requires:
 - GPU (8)
 - TensorFlow
 - Numpy

To evaluate the pre-trained smallNORB checkpoint a GPU is required.
Download the checkpoint from: https://storage.googleapis.com/capsule_toronto/norb_em_checkpoints.tar.gz
Extract into: $HOME
Download the smallNORB tfrecords from: https://storage.googleapis.com/capsule_toronto/smallNORB_data.tar.gz
Extract into: $HOME/smallNORB/
The smallNORB tfrecords can also be generated from norb/norb_convert

To test on smallNORB, with a test error of 1.8:
From the deepest google_research directory, run:
```
python -m capsule_em.experiment  --train=0 --eval_once=1 --eval_size=24300 --ckpnt=$HOME/model.ckpt-1 --final_beta=0.01 --norb_data_dir=$HOME/smallNORB/ --patching=False
```
To get 1.3 (1.4 in the paper), enable patching:
```
python -m capsule_em.experiment  --train=0 --eval_once=1 --eval_size=24300 --ckpnt=$HOME/model.ckpt-1 --final_beta=0.01 --norb_data_dir=$HOME/smallNORB/ --patching=True
```

A docker is now added to the repository. Please install docker with NVIDIA support. Then modify the run_local_gpu.sh with your smallNORB directory (-v option). Running ./run_local_gpu.sh will start training the model.
