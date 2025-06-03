checkpoint_path=$checkpoint_path # ".../logs/FB150k-50/1p.2p.3p.2i.3i-1p.2p.3p.2i.3i.ip.pi/rotate/g-32.0-mode-(Mean,True)-adv-0.5-ngpu-0.1-os-(5000,0,w,u,80)-dataset-(single,3000,e,True,before)-opt-(aggr,adam,cpu,False,5)-sharen-sqrt-lr_step/.../"
pruner_save_path=$pruner_save_path # "$checkpoint_path/pruners/r-1p.2p.3p.2i.3i.pi.ip-b-2i.3i.pi.2pi/.../"
data_path=$data_path # .../FB150k-50
question_path=$question_path # .../webqsp
dataset_name=webqsp


CUDA_VISIBLE_DEVICES=1 python ../search_candidates.py --cuda --do_search --dataset_name $dataset_name \
  --data_path $data_path --question_path $question_path \
  --geo rotate --rotate_mode "(Mean,True)" -d 1600 -g 32 \
  --train_online_mode "(single,0,n,False,before)" --optim_mode "(fast,adam,gpu,True,5)" \
  --checkpoint_path $checkpoint_path --print_on_screen \
  --pruner_save_path $pruner_save_path \
  --max_eps_len_list '(4,5,6,7)' --search_time_stamp "search_results_Feb" --start_idx 0 --end_idx 3000
