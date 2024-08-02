#$ -l tmem=49G
#$ -l h_rt=8:00:00 
#$ -S /bin/bash
#$ -j y
#$ -l gpu=true
#$ -N mil_transformer_uni_regression_10epochs_cv5
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-11.8.source
python3 HistoMIL/Notebooks/mil_run.py --exp-name 'mil_transformer_uni_regression_10epochs_cv5' \
                                      --project-name 'g0-arrest-uni-transformer-regression-cv5' \
                                      --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' \
                                      --task-name 'g0_arrest_score' --pid-name 'PatientID' \
                                      --targets-name 'g0_arrest' 'g0_arrest_score' \
                                      --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                      --split-ratio 0.9 0.1 --step-size 224 \
                                      --precomputed 'uni' \
                                      --mil-algorithm "TransformerRegression" \
                                      --n-epochs 10 \
                                      --monitor-metric 'auroc_val' \
                                      --k-fold 5
echo "Finished at: $(date)"