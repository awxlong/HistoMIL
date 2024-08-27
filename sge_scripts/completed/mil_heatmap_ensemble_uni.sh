#$ -l tmem=16G
#$ -l h_rt=3:00:00 
#$ -S /bin/bash
#$ -j y
#$ -l gpu=true,gpu_type=!(titanxp|titanx)
#$ -N mil_heatmap_ensemble_uni
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-11.8.source
python3 HistoMIL/Notebooks/mil_heatmap.py --exp-name 'mil-inference-transmil-uni' \
                                            --project-name 'g0-arrest-inference-uni-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'uni' \
                                            --mil-algorithm "TransMIL" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'mil_transmil_uni_32epochs_reruncv=0_epoch=10-auroc_val=0.76' 'mil_transmil_uni_32epochs_reruncv=1_epoch=07-auroc_val=0.81' 'mil_transmil_uni_32epochs_reruncv=2_epoch=05-auroc_val=0.78' 'mil_transmil_uni_32epochs_reruncv=3_epoch=08-auroc_val=0.75' 'mil_transmil_uni_32epochs_reruncv=4_epoch=05-auroc_val=0.70'\
                                            --ensemble

python3 HistoMIL/Notebooks/mil_heatmap.py --exp-name 'mil-inference-attentionmil-uni' \
                                            --project-name 'g0-arrest-inference-uni-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'uni' \
                                            --mil-algorithm "AttentionMIL" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'attentionMIL_uni_32epoch_reruncv=0_epoch=31-auroc_val=0.66' 'attentionMIL_uni_32epoch_reruncv=1_epoch=31-auroc_val=0.70' 'attentionMIL_uni_32epoch_reruncv=2_epoch=31-auroc_val=0.70' 'attentionMIL_uni_32epoch_reruncv=3_epoch=28-auroc_val=0.65' 'attentionMIL_uni_32epoch_reruncv=4_epoch=31-auroc_val=0.63'\
                                            --ensemble

python3 HistoMIL/Notebooks/mil_heatmap.py --exp-name 'mil-inference-transformer-uni' \
                                            --project-name 'g0-arrest-inference-uni-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'uni' \
                                            --mil-algorithm "Transformer" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'mil_transformer_uni_8epochs_smaller_0711_0940cv=0_epoch=04-auroc=0.00' 'mil_transformer_uni_8epochs_smaller_0711_0940cv=1_epoch=06-auroc=0.00' 'mil_transformer_uni_8epochs_smaller_0711_0940cv=2_epoch=07-auroc=0.00' 'mil_transformer_uni_8epochs_smaller_0711_0940cv=3_epoch=06-auroc=0.00' 'mil_transformer_uni_8epochs_smaller_0711_0940cv=4_epoch=07-auroc=0.00'\
                                            --ensemble


python3 HistoMIL/Notebooks/mil_heatmap.py --exp-name 'mil-inference-camil-uni' \
                                            --project-name 'g0-arrest-inference-uni-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'uni' \
                                            --mil-algorithm "CAMIL" \
                                            --label-dict "{0:0,1:1}" \
                                            --num-workers 0 \
                                            --ckpt-filenames 'mil_camil_uni_30epochs_cv5_reducelronplateucv=0_epoch=03-auroc_val=0.76' 'mil_camil_uni_30epochs_cv5_reducelronplateucv=1_epoch=07-auroc_val=0.82' 'mil_camil_uni_30epochs_cv5_reducelronplateucv=2_epoch=01-auroc_val=0.76' 'mil_camil_uni_30epochs_cv5_reducelronplateucv=3_epoch=10-auroc_val=0.72' 'mil_camil_uni_30epochs_cv5_reducelronplateucv=4_epoch=07-auroc_val=0.72'\
                                            --ensemble



python3 HistoMIL/Notebooks/mil_heatmap.py --exp-name 'mil-inference-dtfdmil-uni' \
                                            --project-name 'g0-arrest-inference-uni-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'uni' \
                                            --mil-algorithm "DTFD_MIL" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'mil_dtfd_uni_42epochs_cv5_multisteplrcv=0_epoch=41-auroc_val=0.79' 'mil_dtfd_uni_42epochs_cv5_multisteplrcv=1_epoch=35-auroc_val=0.84' 'mil_dtfd_uni_42epochs_cv5_multisteplrcv=2_epoch=24-auroc_val=0.81' 'mil_dtfd_uni_42epochs_cv5_multisteplrcv=3_epoch=33-auroc_val=0.71' 'mil_dtfd_uni_42epochs_cv5_multisteplrcv=4_epoch=15-auroc_val=0.74'\
                                            --ensemble

python3 HistoMIL/Notebooks/mil_heatmap.py --exp-name 'mil-inference-graphtransformer-uni' \
                                            --project-name 'g0-arrest-inference-uni-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'uni' \
                                            --mil-algorithm "GraphTransformer" \
                                            --label-dict "{0:0,1:1}" \
                                            --num-workers 0 \
                                            --ckpt-filenames 'mil_graphtransformer_uni_42epochs_cv5_multisteplrcv=0_epoch=03-auroc_val=0.67' 'mil_graphtransformer_uni_42epochs_cv5_multisteplrcv=1_epoch=36-auroc_val=0.47' 'mil_graphtransformer_uni_42epochs_cv5_multisteplrcv=2_epoch=00-auroc_val=0.72' 'mil_graphtransformer_uni_42epochs_cv5_multisteplrcv=3_epoch=08-auroc_val=0.56' 'mil_graphtransformer_uni_42epochs_cv5_multisteplrcv=4_epoch=05-auroc_val=0.58'\
                                            --ensemble

python3 HistoMIL/Notebooks/mil_heatmap.py --exp-name 'mil-inference-clam-uni' \
                                            --project-name 'g0-arrest-inference-uni-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'uni' \
                                            --mil-algorithm "CLAM" \
                                            --label-dict "{0:0,1:1}" \
                                            --ckpt-filenames 'mil_clam_uni_42epochs_cv5cv=0_epoch=12-auroc_val=0.77' 'mil_clam_uni_42epochs_cv5cv=1_epoch=14-auroc_val=0.83' 'mil_clam_uni_42epochs_cv5cv=2_epoch=05-auroc_val=0.80' 'mil_clam_uni_42epochs_cv5cv=3_epoch=21-auroc_val=0.76' 'mil_clam_uni_42epochs_cv5cv=4_epoch=24-auroc_val=0.73'\
                                            --ensemble

python3 HistoMIL/Notebooks/mil_heatmap.py --exp-name 'mil-inference-transmil-regression-uni' \
                                            --project-name 'g0-arrest-inference-uni-binary' \
                                            --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                            --task-name 'g0_arrest_score' --pid-name 'PatientID' \
                                            --targets-name 'g0_arrest' 'g0_arrest_score' \
                                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                            --split-ratio 0.9 0.1 --step-size 224 \
                                            --precomputed 'uni' \
                                            --mil-algorithm "TransMILRegression" \
                                            --ckpt-filenames 'mil_transmil_uni_regression_32epochs_cv5_rerun_f1_monitorcv=0_epoch=22-auroc_val=0.38' 'mil_transmil_uni_regression_32epochs_cv5_rerun_f1_monitorcv=1_epoch=13-auroc_val=0.24' 'mil_transmil_uni_regression_32epochs_cv5_rerun_f1_monitorcv=2_epoch=21-auroc_val=0.34' 'mil_transmil_uni_regression_32epochs_cv5_rerun_f1_monitorcv=3_epoch=27-auroc_val=0.30' 'mil_transmil_uni_regression_32epochs_cv5_rerun_f1_monitorcv=4_epoch=02-auroc_val=0.39' \
                                            --ensemble

echo "Finished at: $(date)"