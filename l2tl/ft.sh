model_name=finetune_svhn
steps=1200

python finetuning.py \
    --target_dataset=svhn_cropped_small \
    --train_steps=$steps \
    --model_dir=./tmp/${model_name} \
    --train_batch_size=8 \
    --src_num_classes=5 \
    --target_base_learning_rate=0.005 \
    --warm_start_ckpt_path=./tmp/mnist_pretrain/model.ckpt-2000

python evaluate.py \
    --ckpt_path=./tmp/${model_name}/model.ckpt-$steps \
    --src_num_classes=5 \
    --target_dataset=svhn_cropped_small \
    --cls_dense_name=final_dense_dst
