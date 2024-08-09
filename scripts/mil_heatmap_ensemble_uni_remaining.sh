#$ -l tmem=16G
#$ -l h_rt=10:00:00 
#$ -S /bin/bash
#$ -j y
#$ -l gpu=true,gpu_type=!(titanxp|titanx)
#$ -N mil_heatmap_ensemble_uni_remaining_camil_reg
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-11.8.source
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