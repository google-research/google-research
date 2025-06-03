data_path=$data_path # .../FB150k-50
question_path=$question_path # .../webqsp
dataset_name=webqsp

CUDA_VISIBLE_DEVICES=0,2 python ../pretrain_executor.py --do_train --do_test --gpus '0.1' \
 --data_path $data_path --eval_path $question_path --dataset_name $dataset_name \
 -n 128 -b 512 -d 1600 -g 32 \
 -a 0.5 -adv \
 -lr 0.00005 --max_steps 600001 --geo rotate --valid_steps 10000 \
 -rotatem '(Mean,True)' --tasks '1p.2p.3p.2i.3i.ip.pi' --training_tasks '1p.2p.3p.2i.3i' \
 --online_sample --prefix '../logs' --online_sample_mode '(5000,0,w,u,80)' \
 --lr_schedule step \
 --sampler_type sqrt \
 --logit_impl custom \
 --share_negative \
 --share_optim_stats \
 --cpu_num 6 \
 --save_checkpoint_steps 100000 \
 --train_online_mode '(single,3000,e,True,before)' --optim_mode '(aggr,adam,cpu,False,5)' --online_weighted_structure_prob '(1,1,1,1,1)' --print_on_screen \
 --port 29513

