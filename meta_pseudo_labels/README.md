# [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580).

This repository provides the sample implementions of the
[Meta Pseudo Labels](https://arxiv.org/abs/2003.10580) algorithm for two
low-resource image classifciation benchmarks: CIFAR-10-4000 and ImageNet-10%.
The current implementation only runs with TPUs.

To run a CIFAR-10-4000 experiment, please [set up your Cloud TPU environment](https://cloud.google.com/tpu/docs/quickstart) and run the following command:
```
python -m main.py \
  --task_mode="train" \
  --dataset_name="cifar10_4000_mpl" \
  --output_dir="path/to/your/output/dir" \
  --model_type="wrn-28-2" \
  --log_every=100 \
  --master="path/to/your/tpu/worker" \
  --image_size=32 \
  --num_classes=10 \
  --optim_type="momentum" \
  --lr_decay_type="cosine" \
  --save_every=1000 \
  --use_bfloat16 \
  --use_tpu \
  --nouse_augment \
  --reset_output_dir \
  --eval_batch_size=64 \
  --alsologtostderr \
  --running_local_dev \
  --train_batch_size=128 \
  --uda_data=7 \
  --weight_decay=5e-4 \
  --tpu_platform="your_tpu_platform" \
  --tpu_topology="your_tpu_topology" \
  --num_train_steps=300000 \
  --augment_magnitude=16 \
  --batch_norm_batch_size=256 \
  --dense_dropout_rate=0.2 \
  --ema_decay=0.995 \
  --label_smoothing=0.15 \
  --mpl_student_lr_wait_steps=3000 \
  --uda_steps=5000 \
  --uda_temp=0.7 \
  --uda_threshold=0.6 \
  --uda_weight=8
```

To run an ImageNet-10% experiment, please set up your Cloud TPU environment
and run the following command:
```
python -m main.py \
  --task_mode="train" \
  --dataset_name="cifar10_4000_mpl" \
  --output_dir="path/to/your/output/dir" \
  --model_type='resnet-50' \
  --log_every=100 \
  --master="path/to/your/tpu/worker" \
  --image_size=32 \
  --num_classes=10 \
  --optim_type="momentum" \
  --lr_decay_type="cosine" \
  --use_bfloat16 \
  --use_tpu \
  --nouse_augment \
  --reset_output_dir \
  --eval_batch_size=64 \
  --alsologtostderr \
  --running_local_dev \
  --train_batch_size=128 \
  --tpu_platform="your_tpu_platform" \
  --tpu_topology="your_tpu_topology" \
  --label_smoothing=0.1 \
  --grad_bound=5. \
  --uda_data=15 \
  --uda_steps=50000 \
  --uda_temp=0.5 \
  --uda_threshold=0.6 \
  --uda_weight=20. \
  --tpu_platform=tpu_platform \
  --tpu_topology=tpu_topology \
  --train_batch_size=1024 \
  --num_train_steps=700000 \
  --num_warmup_steps=10000 \
  --use_augment=False \
  --augment_magnitude=17 \
  --batch_norm_batch_size=1024 \
  --mpl_student_lr=0.1 \
  --mpl_student_lr_wait_steps=20000 \
  --mpl_student_lr_warmup_steps=5000 \
  --mpl_teacher_lr=0.15 \
  --mpl_teacher_lr_warmup_steps=5000 \
  --ema_decay=0.999 \
  --dense_dropout_rate=0.1 \
  --weight_decay=1e-4 \
  --save_every=1000
```
