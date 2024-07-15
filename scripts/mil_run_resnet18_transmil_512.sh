#$ -l tmem=6G
#$ -l h_rt=36:00:00 
#$ -S /bin/bash
#$ -j y
#$ -l gpu=true
#$ -N mil_transmil_resnet18_512_32epochs
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-11.8.source
python3 HistoMIL/Notebooks/mil_run.py --exp-name 'mil_transmil_resnet18_512_32epochs' \
                                      --project-name 'g0-arrest-resnet18-512-transmil-cv5' \
                                      --wandb-entity-name 'cell-x' --localcohort-name 'COAD' --task-name 'g0_arrest' --pid-name 'PatientID' \
                                      --targets-name 'g0_arrest' \
                                      --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                      --split-ratio 0.9 0.1 \
                                      --step-size 512 \
                                      --precomputed 'resnet18' \
                                      --label-dict "{0:0,1:1}" \
                                      --mil-algorithm "TransMIL" \
                                      --n-epochs 32 \
                                      --monitor-metric 'auroc_val' \
                                      --k-fold 5
echo "Finished at: $(date)"