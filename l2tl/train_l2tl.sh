model_name=l2tl_svhn
steps=1200

python train_l2tl.py \
    --train_batch_size=8 \
    --learning_rate=0.005 \
    --rl_learning_rate=0.01 \
    --target_num_classes=5 \
    --train_steps=$steps \
    --source_train_batch_multiplier=2 \
    --loss_weight_scale=100. \
    --num_choices=100 \
    --first_pretrain_steps=0 \
    --target_val_batch_multiplier=4 \
    --target_train_batch_multiplier=1 \
    --model_dir=./tmp/${model_name} \
    --warm_start_ckpt_path=./tmp/mnist_pretrain/model.ckpt-2000

python evaluate.py \
    --ckpt_path=./tmp/${model_name}/model.ckpt-$steps \
    --target_dataset=svhn_cropped_small \
    --src_num_classes=5 \
    --cls_dense_name=final_target_dense
