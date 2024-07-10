#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=15:00:00 
#$ -S /bin/bash
#$ -j y
#$ -N preprocess_until_patch
cd secrier_lab/persistence/
source /home/xuelonan/secrier_lab/python3.8.5-biomedai.source
source /share/apps/source_files/cuda/cuda-11.8.source
python3 HistoMIL/Notebooks/pre_processing.py --exp-name 'preprocessing' \
                            --project-name 'g0-arrest' \
                            --wandb-entity-name 'cell-x' \
                            --localcohort-name 'COAD' \
                            --task-name 'g0_arrest' \
                            --pid-name 'PatientID' \
                            --targets-name 'g0_arrest' \
                            --cohort-dir '/home/xuelonan/secrier_lab/persistence/' \
                            --split-ratio 0.99 0.01 \
                            --step-size 224 \
                            --label-dict "{'0':0,'1':1}" \
                            --api-dir "/home/xuelonan/secrier_lab/persistence/HistoMIL/" \
                            --concepts-name 'slide' 'tissue' 'patch'
