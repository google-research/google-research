# tumor growth dataset
## zero-shot transfer/data-efficient transfer/standard supervised learning
## zero-shot transfer metrics: dst-zero-shot-encoder_test_rmse_last, dst-zero-shot-decoder_test_rmse_[2-6]-step
## data-efficient transfer metrics: dst-encoder_test_rmse_last, dst-decoder_test_rmse_[2-6]-step
## standard supervised learning metrics: src-encoder_test_rmse_last, src-decoder_test_rmse_[2-6]-step
PYTHONPATH=. python runnables/train_rep_est.py -m +dataset=cancer_sim +target_dataset=cancer_sim +backbone=cotrans_tcn_comp_contrast +backbone/cotrans_tcn_hparams/cancer_sim=\"10\" model.est_head.step_mse_loss_weights_type=inverse exp.tags=231005_tcs_cotrans_invloss exp.seed=10,101,1010,10101,101010

# semi-synthetic MIMIC-III (run zero-shot transfer/data-efficient transfer experiments first)
## zero-shot transfer/data-efficient transfer
## zero-shot transfer metrics: dst-zero-shot-encoder_test_rmse_last, dst-zero-shot-decoder_test_rmse_[2-6]-step
## data-efficient transfer metrics: dst-encoder_test_rmse_last, dst-decoder_test_rmse_[2-6]-step
PYTHONPATH=. python runnables/train_rep_est.py -m +dataset/mimic3_syn_age="0-3_all" +backbone=cotrans_tcn_comp_contrast +backbone/cotrans_tcn_hparams/mimic3_synthetic="all_es" dataset.few_shot_sample_num=0.1 model.est_head.step_mse_loss_weights_type=inverse exp.tags=231005_mimicsyn_cotrans_invloss exp.seed=17,43,44,91,95
## standard supervised learning
## standard supervised learning metrics: src-encoder_test_rmse_last, src-decoder_test_rmse_[2-6]-step
PYTHONPATH=. python runnables/train_rep_est.py -m +dataset/mimic3_syn_age="0-3_all_srctest" +backbone=cotrans_tcn_comp_contrast +backbone/cotrans_tcn_hparams/mimic3_synthetic="all_es" dataset.few_shot_sample_num=0.1 exp.max_epochs=0 exp.finetune_ckpt_type=best model.use_best_model=False exp.seed=17,43,44,91,95 exp.finetune_tag=231005_mimicsyn_cotrans_invloss exp.tags=231005_mimicsyn_cotrans_invloss_srctest exp.skip_train_rep=True

# M5 (run zero-shot transfer/data-efficient transfer experiments first)
## zero-shot transfer/data-efficient transfer
## zero-shot transfer metrics: dst-zero-shot-decoder_test_rmse_[1-6]-step
## data-efficient transfer metrics: dst-decoder_test_rmse_[1-6]-step
PYTHONPATH=. python runnables/train_rep_est.py -m +dataset/m5_category="foods_household_5k" +backbone=cotrans_tcn_comp_contrast +backbone/cotrans_tcn_hparams/m5_real="sales_es" dataset.few_shot_sample_num=0.1 exp.tags=231005_m5_cotrans_tcn_compcontrast exp.seed=10,101,1010,10101,101010
## standard supervised learning
## standard supervised learning metrics: src-decoder_test_rmse_[1-6]-step
PYTHONPATH=. python runnables/train_rep_est.py -m +dataset/m5_category="foods_household_5k_srctest" +backbone=cotrans_tcn_comp_contrast +backbone/cotrans_tcn_hparams/m5_real="sales_es" dataset.few_shot_sample_num=0.1 exp.finetune_tag=231005_m5_cotrans_tcn_compcontrast exp.max_epochs=0 exp.finetune_ckpt_type=best model.use_best_model=False exp.seed=10,101,1010,10101,101010 exp.tags=231005_m5_cotrans_tcn_compcontrast_srctest
