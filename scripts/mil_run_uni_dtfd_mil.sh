#$ -l tmem=9G
#$ -l h_rt=42:00:00 
#$ -S /bin/bash
#$ -j y
#$ -l gpu=true
#$ -N mil_dtfd_uni_200epochs_cv5
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-11.8.source
python3 HistoMIL/Notebooks/mil_run.py --exp-name 'mil_dtfd_uni_200epochs_cv5' \
                                      --project-name 'g0-arrest-uni-dtfdmil-cv5' \
                                      --wandb-entity-name 'anwangxuelong' --localcohort-name 'COAD' --task-name 'g0_arrest' --pid-name 'PatientID' \
                                      --targets-name 'g0_arrest' \
                                      --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                      --split-ratio 0.9 0.1 --step-size 224 \
                                      --precomputed 'uni' \
                                      --label-dict "{0:0,1:1}" \
                                      --mil-algorithm "DTFD-MIL" \
                                      --n-epochs 200 \
                                      --monitor-metric 'auroc_val' \
                                      --k-fold 5
echo "Finished at: $(date)"