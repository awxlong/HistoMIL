#$ -l tmem=1G
#$ -l h_vmem=2G
#$ -l h_rt=00:24:00 
#$ -S /bin/bash
#$ -j y
#$ -N preprocess_clinical
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
python3 HistoMIL/Notebooks/preprocess_clinical.py --exp-name 'preprocessing-clinical' \
                                                  --project-name 'g0-arrest' --wandb-entity-name 'cell-x' \
                                                  --localcohort-name 'COAD' --task-name 'g0_arrest' \
                                                  --pid-name 'PatientID' \
                                                  --targets-name 'g0_arrest' \
                                                  --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                                                  --split-ratio 0.9 0.1 --step-size 224 \
                                                  --label-dict "{0:0,1:1}"  \
                                                  --api-dir '/home/xuelonan/secrier_lab/persistence/HistoMIL/' \
                                                  --k-fold 5

echo "Finished at: $(date)"
