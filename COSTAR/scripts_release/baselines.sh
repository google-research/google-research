# For each setting, metrics are: encoder_test_rmse_last, decoder_test_rmse_[2-6]-step

# tumor growth dataset (run standard supervised learning experiments first)
## standard supervised learning
### MSM
PYTHONPATH=. python runnables/train_msm.py -m +dataset=cancer_sim +target_dataset=cancer_sim +dataset.tr_te_type=src_src +backbone=msm +backbone/msm_hparams/cancer_sim_dseed_10=\"10_tuned\" exp.seed=10,101,1010,10101,101010 exp.tags=231005_msm_tcs_srctest
### RMSN
PYTHONPATH=. python3 runnables/train_rmsn.py -m +dataset=cancer_sim +backbone=rmsn +'backbone/rmsn_hparams/cancer_sim_dseed_10="10_tuned"' exp.seed=10,101,1010,10101,101010 exp.tags="conclude_tcs_cancer_sim_seed_var"
### CRN(ERM)
PYTHONPATH=. python3 runnables/train_enc_dec.py -m +dataset=cancer_sim +backbone=crn_noncausal_troff +'backbone/crn_noncausal_troff_hparams/cancer_sim_dseed_10="10_tuned"' exp.seed=10,101,1010,10101,101010 exp.tags="conclude_tcs_cancer_sim_seed_var"
### CRN
PYTHONPATH=. python3 runnables/train_enc_dec.py -m +dataset=cancer_sim +backbone=crn +'backbone/crn_hparams/cancer_sim_dseed_10="10_tuned"' exp.seed=10,101,1010,10101,101010 exp.tags="conclude_tcs_cancer_sim_seed_var"
### CT(ERM)
PYTHONPATH=. python3 runnables/train_multi.py -m +dataset=cancer_sim +backbone=ct_noncausal_troff +'backbone/ct_noncausal_troff_hparams/cancer_sim_dseed_10="10_tuned"' exp.seed=10,101,1010,10101,101010 exp.tags="conclude_tcs_cancer_sim_seed_var"
### CT
PYTHONPATH=. python3 runnables/train_multi.py -m +dataset=cancer_sim +backbone=ct +'backbone/ct_hparams/cancer_sim_dseed_10="10_tuned"' exp.seed=10,101,1010,10101,101010 exp.tags="conclude_tcs_cancer_sim_seed_var"
### G-Net
PYTHONPATH=. python3 runnables/train_gnet.py -m +dataset=cancer_sim +backbone=gnet +'backbone/gnet_hparams/cancer_sim="10"' exp.seed=10,101,1010,10101,101010 exp.tags="231005_gnet_tcs_10"

## zero-shot transfer
### MSM
PYTHONPATH=. python runnables/train_msm.py -m +dataset=cancer_sim +target_dataset=cancer_sim +dataset.tr_te_type=src_tgt +backbone=msm +backbone/msm_hparams/cancer_sim_dseed_10=\"10_tuned\" exp.seed=10,101,1010,10101,101010 exp.tags=231005_msm_tcs_0shot
### RMSN
PYTHONPATH=. python runnables/train_rmsn.py -m +backbone=rmsn +'backbone/rmsn_hparams/cancer_sim_dseed_10="10_tuned"' +dataset=cancer_sim dataset.coeff=0 +dataset.src_coeff=10 dataset.few_shot_sample_num=100 exp.finetune_tag=conclude_tcs_cancer_sim_seed_var exp.max_epochs=0 exp.finetune_ckpt_type=last exp.seed=10,101,1010,10101,101010 exp.tags=231005_bs_tcs_10_0shot
### CRN(ERM)
PYTHONPATH=. python3 runnables/train_enc_dec.py -m +dataset=cancer_sim +backbone=crn_noncausal_troff +'backbone/crn_noncausal_troff_hparams/cancer_sim_dseed_10="10_tuned"' dataset.coeff=0 +dataset.src_coeff=10 dataset.few_shot_sample_num=100 exp.finetune_tag=conclude_tcs_cancer_sim_seed_var exp.max_epochs=0 exp.finetune_ckpt_type=last exp.seed=10,101,1010,10101,101010 exp.tags=231005_bs_tcs_10_0shot
### CRN
PYTHONPATH=. python3 runnables/train_enc_dec.py -m +dataset=cancer_sim +backbone=crn +'backbone/crn_hparams/cancer_sim_dseed_10="10_tuned"' dataset.coeff=0 +dataset.src_coeff=10 dataset.few_shot_sample_num=100 exp.finetune_tag=conclude_tcs_cancer_sim_seed_var exp.max_epochs=0 exp.finetune_ckpt_type=last exp.seed=10,101,1010,10101,101010 exp.tags=231005_bs_tcs_10_0shot
### CT(ERM)
PYTHONPATH=. python3 runnables/train_multi.py -m +dataset=cancer_sim +backbone=ct_noncausal_troff +'backbone/ct_noncausal_troff_hparams/cancer_sim_dseed_10="10_tuned"' dataset.coeff=0 +dataset.src_coeff=10 dataset.few_shot_sample_num=100 exp.finetune_tag=conclude_tcs_cancer_sim_seed_var exp.max_epochs=0 exp.finetune_ckpt_type=last exp.seed=10,101,1010,10101,101010 exp.tags=231005_bs_tcs_10_0shot
### CT
PYTHONPATH=. python3 runnables/train_multi.py -m +dataset=cancer_sim +backbone=ct +'backbone/ct_hparams/cancer_sim_dseed_10="10_tuned"' dataset.coeff=0 +dataset.src_coeff=10 dataset.few_shot_sample_num=100 exp.finetune_tag=conclude_tcs_cancer_sim_seed_var exp.max_epochs=0 exp.finetune_ckpt_type=last exp.seed=10,101,1010,10101,101010 exp.tags=231005_bs_tcs_10_0shot
### G-Net
PYTHONPATH=. python3 runnables/train_gnet.py -m +dataset=cancer_sim +backbone=gnet +'backbone/gnet_hparams/cancer_sim="10"' dataset.coeff=0 dataset.few_shot_sample_num=100 exp.finetune_tag=231005_gnet_tcs_10 exp.finetune_ckpt_type=last exp.max_epochs=0 exp.seed=10,101,1010,10101,101010 exp.tags="231005_gnet_tcs_10_0shot"

## data-efficient transfer
### MSM
PYTHONPATH=. python runnables/train_msm.py -m +dataset=cancer_sim +target_dataset=cancer_sim +dataset.tr_te_type=srctgt_tgt +backbone=msm +backbone/msm_hparams/cancer_sim_dseed_10=\"10_tuned\" exp.seed=10,101,1010,10101,101010 exp.tags=231005_msm_tcs_fewshot
### RMSN
PYTHONPATH=. python runnables/train_rmsn.py -m +backbone=rmsn +'backbone/rmsn_hparams/cancer_sim_dseed_10="10_tuned"' +dataset=cancer_sim dataset.coeff=0 +dataset.src_coeff=10 dataset.few_shot_sample_num=100 exp.finetune_tag=conclude_tcs_cancer_sim_seed_var_rmsn exp.finetune_ckpt_type=last model.use_best_model=True exp.seed=10,101,1010,10101,101010 exp.tags=231005_bs_tcs_10_fewshot
### CRN(ERM)
PYTHONPATH=. python3 runnables/train_enc_dec.py -m +dataset=cancer_sim +backbone=crn_noncausal_troff +'backbone/crn_noncausal_troff_hparams/cancer_sim_dseed_10="10_tuned"' dataset.coeff=0 +dataset.src_coeff=10 dataset.few_shot_sample_num=100 exp.finetune_tag=conclude_tcs_cancer_sim_seed_var exp.finetune_ckpt_type=last model.use_best_model=True exp.seed=10,101,1010,10101,101010 exp.tags=231005_bs_tcs_10_fewshot
### CRN
PYTHONPATH=. python3 runnables/train_enc_dec.py -m +dataset=cancer_sim +backbone=crn +'backbone/crn_hparams/cancer_sim_dseed_10="10_tuned"' dataset.coeff=0 +dataset.src_coeff=10 dataset.few_shot_sample_num=100 exp.finetune_tag=conclude_tcs_cancer_sim_seed_var exp.finetune_ckpt_type=last model.use_best_model=True exp.seed=10,101,1010,10101,101010 exp.tags=231005_bs_tcs_10_fewshot
### CT(ERM)
PYTHONPATH=. python3 runnables/train_multi.py -m +dataset=cancer_sim +backbone=ct_noncausal_troff +'backbone/ct_noncausal_troff_hparams/cancer_sim_dseed_10="10_tuned"' dataset.coeff=0 +dataset.src_coeff=10 dataset.few_shot_sample_num=100 exp.finetune_tag=conclude_tcs_cancer_sim_seed_var exp.finetune_ckpt_type=last model.use_best_model=True exp.seed=10,101,1010,10101,101010 exp.tags=231005_bs_tcs_10_fewshot
### CT
PYTHONPATH=. python3 runnables/train_multi.py -m +dataset=cancer_sim +backbone=ct +'backbone/ct_hparams/cancer_sim_dseed_10="10_tuned"' dataset.coeff=0 +dataset.src_coeff=10 dataset.few_shot_sample_num=100 exp.finetune_tag=conclude_tcs_cancer_sim_seed_var exp.finetune_ckpt_type=last model.use_best_model=True exp.seed=10,101,1010,10101,101010 exp.tags=231005_bs_tcs_10_fewshot
### G-Net
PYTHONPATH=. python3 runnables/train_gnet.py -m +dataset=cancer_sim +backbone=gnet +'backbone/gnet_hparams/cancer_sim="10"' dataset.coeff=0 dataset.few_shot_sample_num=100 model.use_best_model=True exp.finetune_tag=231005_gnet_tcs_10 exp.finetune_ckpt_type=last exp.seed=10,101,1010,10101,101010 exp.tags="231005_gnet_tcs_10_fewshot"


# semi-synthetic MIMIC-III (run zero-shot transfer experiments first)
## zero-shot transfer
### MSM
PYTHONPATH=. python runnables/train_msm.py -m +dataset/mimic3_syn_age="0-3_all" +dataset.tr_te_type=src_tgt +backbone=msm +'backbone/msm_hparams/mimic3_synthetic="all_es"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 exp.tags=231005_msm_mimicsyn_0shot
### RMSN
PYTHONPATH=. python runnables/train_rmsn.py -m  +dataset/mimic3_syn_age="0-3_all" +backbone=rmsn +'backbone/rmsn_hparams/mimic3_synthetic="all"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 exp.tags=231005_mimicsyn_all_pretrain
### CRN(ERM)
PYTHONPATH=. python runnables/train_enc_dec.py -m +dataset/mimic3_syn_age="0-3_all" +backbone=crn_noncausal_troff +'backbone/crn_noncausal_troff_hparams/mimic3_synthetic="all"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 exp.tags=231005_mimicsyn_all_pretrain
### CRN
PYTHONPATH=. python runnables/train_enc_dec.py -m +dataset/mimic3_syn_age="0-3_all" +backbone=crn +'backbone/crn_hparams/mimic3_synthetic="all_es"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 exp.tags=231005_mimicsyn_all_pretrain
### CT(ERM)
PYTHONPATH=. python runnables/train_multi.py -m +dataset/mimic3_syn_age="0-3_all" +backbone=ct_noncausal_troff +'backbone/ct_noncausal_troff_hparams/mimic3_synthetic="all"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 exp.tags=231005_mimicsyn_all_pretrain
### CT
PYTHONPATH=. python runnables/train_multi.py -m +dataset/mimic3_syn_age="0-3_all" +backbone=ct +'backbone/ct_hparams/mimic3_synthetic="all_es"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 exp.tags=231005_mimicsyn_all_pretrain
### G-Net
PYTHONPATH=. python runnables/train_gnet.py -m +dataset/mimic3_syn_age="0-3_all" +backbone=gnet +'backbone/gnet_hparams/mimic3_synthetic="all_es"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 exp.tags=231005_mimicsyn_all_gnet_pretrain

## data-efficient transfer
### MSM
PYTHONPATH=. python runnables/train_msm.py -m +dataset/mimic3_syn_age="0-3_all" +dataset.tr_te_type=srctgt_tgt +backbone=msm +'backbone/msm_hparams/mimic3_synthetic="all_es"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 exp.tags=231005_msm_mimicsyn_fewshot
### RMSN
PYTHONPATH=. python runnables/train_rmsn.py -m +dataset/mimic3_syn_age="0-3_all" +backbone=rmsn +'backbone/rmsn_hparams/mimic3_synthetic="all"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=True exp.finetune_tag="231005_mimicsyn_all_pretrain" exp.tags=231005_mimicsyn_all_fewshot
### CRN(ERM)
PYTHONPATH=. python runnables/train_enc_dec.py -m +dataset/mimic3_syn_age="0-3_all" +backbone=crn_noncausal_troff +'backbone/crn_noncausal_troff_hparams/mimic3_synthetic="all"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=True exp.finetune_tag="231005_mimicsyn_all_pretrain" exp.tags=231005_mimicsyn_all_fewshot
### CRN
PYTHONPATH=. python runnables/train_enc_dec.py -m +dataset/mimic3_syn_age="0-3_all" +backbone=crn +'backbone/crn_hparams/mimic3_synthetic="all"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=True exp.finetune_tag="231005_mimicsyn_all_pretrain" exp.tags=231005_mimicsyn_all_fewshot
### CT(ERM)
PYTHONPATH=. python runnables/train_multi.py -m +dataset/mimic3_syn_age="0-3_all" +backbone=ct_noncausal_troff +'backbone/ct_noncausal_troff_hparams/mimic3_synthetic="all"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=True exp.finetune_tag="231005_mimicsyn_all_pretrain" exp.tags=231005_mimicsyn_all_fewshot
### CT
PYTHONPATH=. python runnables/train_multi.py -m +dataset/mimic3_syn_age="0-3_all" +backbone=ct +'backbone/ct_hparams/mimic3_synthetic="all"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=True exp.finetune_tag="231005_mimicsyn_all_pretrain" exp.tags=231005_mimicsyn_all_fewshot
### G-Net
PYTHONPATH=. python runnables/train_gnet.py -m +dataset/mimic3_syn_age="0-3_all" +backbone=gnet +'backbone/gnet_hparams/mimic3_synthetic="all_es"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=True exp.finetune_tag="231005_mimicsyn_all_gnet_pretrain" exp.tags=231005_mimicsyn_all_gnet_fewshot

## standard supervised learning
### MSM
PYTHONPATH=. python runnables/train_msm.py -m +dataset/mimic3_syn_age="0-3_all_srctest" +dataset.tr_te_type=src_src +backbone=msm +'backbone/msm_hparams/mimic3_synthetic="all_es"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 exp.tags=231005_msm_mimicsyn_srctest
### RMSN
PYTHONPATH=. python runnables/train_rmsn.py -m +dataset/mimic3_syn_age="0-3_all_srctest" +backbone=rmsn +'backbone/rmsn_hparams/mimic3_synthetic="all"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=False exp.finetune_tag="231005_mimicsyn_all_pretrain" exp.max_epochs=0 exp.finetune_ckpt_type=best exp.tags=231005_mimicsyn_all_srctest
### CRN(ERM)
PYTHONPATH=. python runnables/train_enc_dec.py -m +dataset/mimic3_syn_age="0-3_all_srctest" +backbone=crn_noncausal_troff +'backbone/crn_noncausal_troff_hparams/mimic3_synthetic="all"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=False exp.finetune_tag="231005_mimicsyn_all_pretrain" exp.max_epochs=0 exp.finetune_ckpt_type=best exp.tags=231005_mimicsyn_all_srctest
### CRN
PYTHONPATH=. python runnables/train_enc_dec.py -m +dataset/mimic3_syn_age="0-3_all_srctest" +backbone=crn +'backbone/crn_hparams/mimic3_synthetic="all"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=False exp.finetune_tag="231005_mimicsyn_all_pretrain" exp.max_epochs=0 exp.finetune_ckpt_type=best exp.tags=231005_mimicsyn_all_srctest
### CT(ERM)
PYTHONPATH=. python runnables/train_multi.py -m +dataset/mimic3_syn_age="0-3_all_srctest" +backbone=ct_noncausal_troff +'backbone/ct_noncausal_troff_hparams/mimic3_synthetic="all"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=False exp.finetune_tag="231005_mimicsyn_all_pretrain" exp.max_epochs=0 exp.finetune_ckpt_type=best exp.tags=231005_mimicsyn_all_srctest
### CT
PYTHONPATH=. python runnables/train_multi.py -m +dataset/mimic3_syn_age="0-3_all_srctest" +backbone=ct +'backbone/ct_hparams/mimic3_synthetic="all"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=False exp.finetune_tag="231005_mimicsyn_all_pretrain" exp.max_epochs=0 exp.finetune_ckpt_type=best exp.tags=231005_mimicsyn_all_srctest
### G-Net
PYTHONPATH=. python runnables/train_gnet.py -m +dataset/mimic3_syn_age="0-3_all_srctest" +backbone=gnet +'backbone/gnet_hparams/mimic3_synthetic="all_es"' exp.seed=17,43,44,91,95 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=False exp.finetune_tag="231005_mimicsyn_all_gnet_pretrain" exp.max_epochs=0 exp.finetune_ckpt_type=best exp.tags=231005_mimicsyn_all_gnet_srctest


# M5 (run zero-shot transfer experiments first)
## zero-shot transfer
### MSM
PYTHONPATH=. python runnables/train_msm.py -m +backbone=msm +'backbone/msm_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k" +dataset.tr_te_type=src_tgt exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 exp.tags=231005_msm_m5_0shot
### RMSN
PYTHONPATH=. python runnables/train_rmsn.py -m +backbone=rmsn +'backbone/rmsn_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 exp.tags=231005_m5_foods_household_5k_es
### CRN(ERM)
PYTHONPATH=. python runnables/train_enc_dec.py -m +backbone=crn_noncausal_troff +'backbone/crn_noncausal_troff_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 exp.tags=231005_m5_foods_household_5k_es
### CRN
PYTHONPATH=. python runnables/train_enc_dec.py -m +backbone=crn +'backbone/crn_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 exp.tags=231005_m5_foods_household_5k_es
### CT(ERM)
PYTHONPATH=. python runnables/train_multi.py -m +backbone=ct_noncausal_troff +'backbone/ct_noncausal_troff_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 exp.tags=231005_m5_foods_household_5k_es
### CT
PYTHONPATH=. python runnables/train_multi.py -m +backbone=ct +'backbone/ct_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 exp.tags=231005_m5_foods_household_5k_es
### G-Net
PYTHONPATH=. python runnables/train_gnet.py -m +backbone=gnet +'backbone/gnet_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 exp.tags=231005_m5_gnet_pretrain

## data-efficient transfer
### MSM
PYTHONPATH=. python runnables/train_msm.py -m +backbone=msm +'backbone/msm_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k" +dataset.tr_te_type=srctgt_tgt exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 exp.tags=231005_msm_m5_fewshot
### RMSN
PYTHONPATH=. python runnables/train_rmsn.py -m +backbone=rmsn +'backbone/rmsn_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=True exp.finetune_tag=231005_m5_foods_household_5k_es exp.tags=231005_m5_foods_household_5k_es_fewshot
### CRN(ERM)
PYTHONPATH=. python runnables/train_enc_dec.py -m +backbone=crn_noncausal_troff +'backbone/crn_noncausal_troff_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=True exp.finetune_tag=231005_m5_foods_household_5k_es exp.tags=231005_m5_foods_household_5k_es_fewshot
### CRN
PYTHONPATH=. python runnables/train_enc_dec.py -m +backbone=crn +'backbone/crn_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=True exp.finetune_tag=231005_m5_foods_household_5k_es exp.tags=231005_m5_foods_household_5k_es_fewshot
### CT(ERM)
PYTHONPATH=. python runnables/train_multi.py -m +backbone=ct_noncausal_troff +'backbone/ct_noncausal_troff_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=True exp.finetune_tag=231005_m5_foods_household_5k_es exp.tags=231005_m5_foods_household_5k_es_fewshot
### CT
PYTHONPATH=. python runnables/train_multi.py -m +backbone=ct +'backbone/ct_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=True exp.finetune_tag=231005_m5_foods_household_5k_es exp.tags=231005_m5_foods_household_5k_es_fewshot
### G-Net
PYTHONPATH=. python runnables/train_gnet.py -m +backbone=gnet +'backbone/gnet_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=True exp.finetune_tag=231005_m5_gnet_pretrain exp.tags=231005_m5_gnet_fewshot

## standard supervised learning
### MSM
PYTHONPATH=. python runnables/train_msm.py -m +backbone=msm +'backbone/msm_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k_srctest" +dataset.tr_te_type=src_src exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 exp.tags=231005_msm_m5_srctest
### RMSN
PYTHONPATH=. python runnables/train_rmsn.py -m +backbone=rmsn +'backbone/rmsn_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k_srctest" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=False exp.finetune_tag=231005_m5_foods_household_5k_es exp.max_epochs=0 exp.finetune_ckpt_type=best exp.tags=231005_m5_foods_household_5k_es_srctest
### CRN(ERM)
PYTHONPATH=. python runnables/train_enc_dec.py -m +backbone=crn_noncausal_troff +'backbone/crn_noncausal_troff_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k_srctest" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=False exp.finetune_tag=231005_m5_foods_household_5k_es exp.max_epochs=0 exp.finetune_ckpt_type=best exp.tags=231005_m5_foods_household_5k_es_srctest
### CRN
PYTHONPATH=. python runnables/train_enc_dec.py -m +backbone=crn +'backbone/crn_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k_srctest" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=False exp.finetune_tag=231005_m5_foods_household_5k_es exp.max_epochs=0 exp.finetune_ckpt_type=best exp.tags=231005_m5_foods_household_5k_es_srctest
### CT(ERM)
PYTHONPATH=. python runnables/train_multi.py -m +backbone=ct_noncausal_troff +'backbone/ct_noncausal_troff_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k_srctest" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=False exp.finetune_tag=231005_m5_foods_household_5k_es exp.max_epochs=0 exp.finetune_ckpt_type=best exp.tags=231005_m5_foods_household_5k_es_srctest
### CT
PYTHONPATH=. python runnables/train_multi.py -m +backbone=ct +'backbone/ct_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k_srctest" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=False exp.finetune_tag=231005_m5_foods_household_5k_es exp.max_epochs=0 exp.finetune_ckpt_type=best exp.tags=231005_m5_foods_household_5k_es_srctest
### G-Net
PYTHONPATH=. python runnables/train_gnet.py -m +backbone=gnet +'backbone/gnet_hparams/m5_real="sales_es"' +dataset/m5_category="foods_household_5k_srctest" exp.seed=10,101,1010,10101,101010 dataset.few_shot_sample_num=0.1 dataset.use_few_shot=True model.use_best_model=False exp.finetune_tag=231005_m5_gnet_pretrain exp.max_epochs=0 exp.finetune_ckpt_type=best exp.tags=231005_m5_gnet_srctest
