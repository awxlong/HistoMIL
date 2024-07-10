#$ -l tmem=3G
#$ -l h_rt=168:00:00 
#$ -S /bin/bash
#$ -j y
#$ -l gpu=true
#$ -N mil_attentionMIL_uni_32epochs
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-11.8.source
python3 HistoMIL/Notebooks/mil_run.py --exp-name 'attentionMIL_uni_32epoch' \
                                      --project-name 'g0-arrest-uni-attentionMIL' \
                                      --wandb-entity-name 'cell-x' --localcohort-name 'COAD' --task-name 'g0_arrest' --pid-name 'PatientID' \
                                      --targets-name 'g0_arrest' \
                                      --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                      --split-ratio 0.8 0.2 --step-size 224 --precomputed 'uni' \
                                      --label-dict "{0:0,1:1}" \
                                      --mil-algorithm "AttentionMIL" \
                                      --n-epochs 32 \
                                      --monitor-metric 'auroc_val' \
                                      --k-fold 3
echo "Finished at: $(date)"