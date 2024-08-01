#$ -l tmem=12G
#$ -l h_rt=32:00:00 
#$ -S /bin/bash
#$ -j y
#$ -l gpu=true,gpu_type=!(titanxp|titanx|gtx1080ti|rtx2080ti)
#$ -N mil_transmil_uni_regression_32epochs_cv5_rerun_f1_monitor
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-11.8.source
python3 HistoMIL/Notebooks/mil_run.py --exp-name 'mil_transmil_uni_regression_32epochs_cv5_rerun_f1_monitor' \
                                      --project-name 'g0-arrest-uni-transmil-regression-cv5' \
                                      --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                      --task-name 'g0_arrest_score' --pid-name 'PatientID' \
                                      --targets-name 'g0_arrest' 'g0_arrest_score' \
                                      --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                      --split-ratio 0.9 0.1 --step-size 224 \
                                      --precomputed 'uni' \
                                      --mil-algorithm "TransMILRegression" \
                                      --n-epochs 32 \
                                      --monitor-metric 'f1_val' \
                                      --k-fold 5
echo "Finished at: $(date)"